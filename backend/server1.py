import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from scipy.signal import butter, sosfilt, sosfilt_zi

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import uvicorn


# ============================================================
# Configuration
# ============================================================

NETWORK = "UO"
STATION_PATTERN = "DUCK*"
CHANNEL = "HNZ"

SEEDLINK_SERVER = "rtserve.earthscope.org:18000"
SEEDLINK_TIMEOUT = 30

FDSN_STATION_URL = (
    "https://service.earthscope.org/"
    "fdsnws/station/1/query"
)

WINDOW_LENGTH = 30.0
RMS_WINDOW = 5.0

FILTER_LOW = 1.0
FILTER_HIGH = 10.0
FILTER_ORDER = 4

UPDATE_INTERVAL = 0.20

BUFFER_SAMPLES = 6000

FILTER_SAMPLE_RATE = 200.0


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_FILE = BASE_DIR / "index.html"


# ============================================================
# Filter
# ============================================================

sos = butter(
    FILTER_ORDER,
    [FILTER_LOW, FILTER_HIGH],
    btype="bandpass",
    fs=FILTER_SAMPLE_RATE,
    output="sos",
)


# ============================================================
# Station metadata
# ============================================================

station_metadata = {}

metadata_lock = threading.Lock()


def fetch_station_metadata():

    params = {
        "network": NETWORK,
        "station": STATION_PATTERN,
        "level": "station",
        "format": "text",
        "nodata": "404",
    }

    url = (
        FDSN_STATION_URL
        + "?"
        + urlencode(params)
    )

    print()
    print("Querying EarthScope FDSN station metadata...")
    print(url)

    try:

        with urlopen(
            url,
            timeout=30
        ) as response:

            text = (
                response
                .read()
                .decode("utf-8")
            )

    except Exception as e:

        print(
            "Could not retrieve station metadata:"
        )

        print(
            repr(e)
        )

        return {}

    stations = {}

    lines = text.strip().splitlines()

    for line in lines:

        if not line.strip():
            continue

        if line.startswith("#"):
            continue

        parts = line.split("|")

        if len(parts) < 8:
            continue

        try:

            network = parts[0].strip()

            code = parts[1].strip()

            latitude = float(parts[2])

            longitude = float(parts[3])

            elevation = float(parts[4])

            site_name = parts[5].strip()

            start_time = parts[6].strip()

            end_time = parts[7].strip()

        except (
            ValueError,
            IndexError
        ):

            continue

        if network != NETWORK:
            continue

        if not code.startswith("DUCK"):
            continue

        # Ignore stations whose metadata has ended.

        if end_time:

            try:

                end_dt = datetime.fromisoformat(
                    end_time.replace(
                        "Z",
                        "+00:00"
                    )
                )

                if end_dt.tzinfo is None:

                    end_dt = (
                        end_dt.replace(
                            tzinfo=timezone.utc
                        )
                    )

                if (
                    end_dt.timestamp()
                    < time.time()
                ):

                    continue

            except Exception:

                pass

        stations[code] = {

            "code": code,

            "network": network,

            "latitude": latitude,

            "longitude": longitude,

            "elevation": elevation,

            "site_name": site_name,

            "start_time": start_time,

            "end_time": end_time,

        }

    print(
        f"Found {len(stations)} active "
        "DuckQuake station(s)."
    )

    for code in sorted(stations):

        station = stations[code]

        print(
            f"  {code}: "
            f"{station['latitude']:.6f}, "
            f"{station['longitude']:.6f}"
        )

    return stations


# ============================================================
# Discover stations
# ============================================================

station_metadata = fetch_station_metadata()


if not station_metadata:

    print()

    print(
        "WARNING: No DuckQuake stations "
        "were found in FDSN metadata."
    )

    print()


# ============================================================
# Dynamic waveform buffers
# ============================================================

buffers = {}

times = {}

locks = {}

# Persistent IIR filter state for each station.
#
# This allows the filter to continue seamlessly
# from one SeedLink packet to the next.

filter_states = {}


def initialize_station(
    station
):

    if station in buffers:
        return

    buffers[station] = deque(
        maxlen=BUFFER_SAMPLES
    )

    times[station] = deque(
        maxlen=BUFFER_SAMPLES
    )

    locks[station] = threading.Lock()

    # No filter state until the first packet arrives.
    filter_states[station] = None


for station in station_metadata:

    initialize_station(
        station
    )


# ============================================================
# SeedLink
# ============================================================

class DuckQuakeClient(
    EasySeedLinkClient
):

    def on_data(
        self,
        trace
    ):

        station = (
            trace.stats.station
        )

        if station not in buffers:

            initialize_station(
                station
            )

        if len(trace.data) == 0:
            return

        data = np.asarray(
            trace.data,
            dtype=float
        )

        # Replace invalid samples.
        data[
            ~np.isfinite(data)
        ] = 0.0

        # ----------------------------------------------------
        # Stateful causal filtering
        # ----------------------------------------------------
        #
        # Unlike sosfiltfilt(), this filter:
        #
        #   1. operates causally
        #   2. does not require future samples
        #   3. preserves filter state between packets
        #
        # This prevents each SeedLink packet from having
        # independent filter startup/edge effects.
        # ----------------------------------------------------

        try:

            if filter_states[station] is None:

                # Initialize the filter in its steady state
                # based on the first sample.

                zi = (
                    sosfilt_zi(sos)
                    * data[0]
                )

                data, filter_states[station] = sosfilt(
                    sos,
                    data,
                    zi=zi
                )

            else:

                data, filter_states[station] = sosfilt(
                    sos,
                    data,
                    zi=filter_states[station]
                )

        except Exception as e:

            print(
                f"Filter error for "
                f"{station}: {e}"
            )

            return

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        start = (
            trace.stats.starttime
        )

        sample_rate = float(
            trace.stats.sampling_rate
        )

        dt = (
            1.0 /
            sample_rate
        )

        # ----------------------------------------------------
        # Append to continuous station buffer
        # ----------------------------------------------------

        with locks[station]:

            for i, value in enumerate(
                data
            ):

                t = float(
                    start +
                    i * dt
                )

                times[
                    station
                ].append(t)

                buffers[
                    station
                ].append(
                    float(value)
                )


def start_seedlink():

    if not station_metadata:

        print(
            "SeedLink not started because "
            "no stations were discovered."
        )

        return

    try:

        print()

        print(
            "Starting DuckQuake "
            "SeedLink client..."
        )

        client = DuckQuakeClient(
            SEEDLINK_SERVER,
            autoconnect=False
        )

        client.conn.timeout = (
            SEEDLINK_TIMEOUT
        )

        print(
            "Connecting to EarthScope "
            f"SeedLink ({SEEDLINK_SERVER})..."
        )

        client.connect()

        print(
            "Connected to EarthScope SeedLink."
        )

        for station in sorted(
            station_metadata
        ):

            print(
                f"Selecting "
                f"{NETWORK}.{station}.."
                f"{CHANNEL}"
            )

            client.select_stream(
                NETWORK,
                station,
                CHANNEL
            )

        print(
            "Selected all DuckQuake stations."
        )

        print(
            "Starting waveform stream..."
        )

        client.run()

    except Exception as e:

        print()

        print(
            "SeedLink connection failed:"
        )

        print(
            repr(e)
        )

        print()

        print(
            "FastAPI is still running."
        )


seedlink_thread = threading.Thread(
    target=start_seedlink,
    daemon=True
)

seedlink_thread.start()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="DuckQuake API"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# Frontend
# ============================================================

@app.get("/")
def frontend():

    return FileResponse(
        INDEX_FILE
    )


# ============================================================
# Station metadata
# ============================================================

@app.get("/stations")
def get_stations():

    with metadata_lock:

        return {
            "stations":
                list(
                    station_metadata.values()
                )
        }


# ============================================================
# Station waveform data
# ============================================================

def get_station_data(
    station
):

    if station not in buffers:
        return None

    with locks[station]:

        if len(
            buffers[station]
        ) == 0:

            return None

        t = np.asarray(
            times[station],
            dtype=float
        )

        data = np.asarray(
            buffers[station],
            dtype=float
        )

    if len(data) < 10:
        return None

    current_time = t[-1]

    waveform_start = (
        current_time -
        WINDOW_LENGTH
    )

    waveform_mask = (
        t >= waveform_start
    )

    waveform_t = (
        t[waveform_mask]
    )

    waveform_data = (
        data[waveform_mask]
    )

    rms_start = (
        current_time -
        RMS_WINDOW
    )

    rms_mask = (
        t >= rms_start
    )

    rms_data = (
        data[rms_mask]
    )

    if len(rms_data) > 0:

        rms = float(
            np.sqrt(
                np.mean(
                    rms_data ** 2
                )
            )
        )

    else:

        rms = 0.0

    # Limit data sent to browser.

    MAX_POINTS = 1200

    if len(
        waveform_data
    ) > MAX_POINTS:

        indices = np.linspace(
            0,
            len(waveform_data) - 1,
            MAX_POINTS
        ).astype(int)

        waveform_t = (
            waveform_t[indices]
        )

        waveform_data = (
            waveform_data[indices]
        )

    return {

        "time":
            current_time,

        "rms":
            rms,

        "waveform_time":
            waveform_t.tolist(),

        "waveform":
            waveform_data.tolist(),

    }


# ============================================================
# Debug/status endpoint
# ============================================================

@app.get("/status")
def status():

    result = {}

    for station in sorted(
        station_metadata
    ):

        if station not in buffers:

            result[station] = {

                "samples": 0,

                "latest_time": None,

            }

            continue

        with locks[station]:

            n = len(
                buffers[station]
            )

            if n > 0:

                latest_time = float(
                    times[
                        station
                    ][-1]
                )

            else:

                latest_time = None

        result[station] = {

            "samples": n,

            "latest_time":
                latest_time,

        }

    return result


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print(
        "Browser connected to /ws"
    )

    try:

        while True:

            payload = {

                "time": None,

                "stations": {}

            }

            station_times = []

            for station in sorted(
                station_metadata
            ):

                result = (
                    get_station_data(
                        station
                    )
                )

                if result is not None:

                    payload[
                        "stations"
                    ][station] = result

                    station_times.append(
                        result["time"]
                    )

            if station_times:

                payload["time"] = max(
                    station_times
                )

            await websocket.send_json(
                payload
            )

            await asyncio.sleep(
                UPDATE_INTERVAL
            )

    except Exception as e:

        print(
            f"WebSocket disconnected: {e}"
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    # Locally:
    #
    #     python backend/server.py
    #
    # Online:
    #
    #     Render runs:
    #     uvicorn backend.server:app
    #
    # $PORT is supplied by Render.

    import os

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
