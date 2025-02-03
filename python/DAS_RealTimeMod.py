# Module containing real-time processing classes and functions
import numpy as np
from obspy.core.trace import Trace 
from obspy.core.trace import Stats
from obspy.core.utcdatetime import UTCDateTime
import os

# Necessary for multi-threading 
import asyncio

# Necessary to run PhaseNet-DAS in real time
import threading
import dateutil.parser
from datetime import datetime, timezone
import atexit
import pandas as pd
# DAS utilities related to picking process
import DAS_ML


############################################################################################################
# DAS RingBuffer
############################################################################################################

time_format = "%Y-%m-%dT%H%M%SZ"

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self, buff_size, good_ch):
        """
        Input:
        buff_size [int]: maximum number of time samples in the ring buffer  
        """
        if buff_size < 0:
            raise ValueError("buff_size must be a positive integer")
        self.max = buff_size
        self.data = []
        self.good_ch = good_ch
        self.channels_info = None
        self.timeStamps = [] # Rolling buffer of timestamp axis

    class __Full:
            """ class that implements a full buffer """
            def append(self, x, timestamps=None):
                """ Append an element overwriting the oldest one. """
                self.data[self.cur] = np.expand_dims(x[self.good_ch], axis=1)
                if timestamps is not None:
                    self.timeStamps[self.cur] = timestamps
                self.cur = (self.cur+1) % self.max
            
            def getData(self):
                """ Return array of elements in correct order """
                return np.concatenate(self.data[self.cur:]+self.data[:self.cur], axis=1)
            
            def getTimeStamps(self):
                """ Return array of timestamps in correct order """
                return np.array(self.timeStamps[self.cur:]+self.timeStamps[:self.cur])
            
            def writeObsPyTraces(self, fs, datapath, scaling=1e6):
                """Method to write Obspy traces"""
                if self.channels_info is None:
                    raise ValueError("Call method setObspyTraceHeader to set trace info before using writeObsPyTraces")
                # Getting channel data to write and timestamps
                traceData = (self.getData()[self.chIds,:]*scaling).astype(np.int32)
                timeStamps = self.getTimeStamps()
                # Loop for writing traces
                for idx in range(len(self.chIds)):
                    self.stats[idx].sampling_rate = fs
                    self.stats[idx].npts = traceData.shape[1]
                    self.stats[idx].starttime = UTCDateTime(timeStamps[0])
                    self.stats[idx].delta = 1.0/fs
                    tr = Trace(traceData[idx,:], header=self.stats[idx])
                    file_path = "%s/%s_%s.mseed"%(datapath,self.channels_info[idx], timeStamps[0].strftime(time_format))
                    tr.write(file_path, format="MSEED", reclen=512, encoding='STEIM2')
                    os.chmod(file_path, 0o644)
                return
            
            def send2ew(self, fs, waveMod, ringID=0, scaling=1e6):
                """
                    Sends buffered seismic data to Earthworm wavering.

                    Parameters:
                        - fs (float): Sampling frequency of the data.
                        - waveMod (object): Earthworm wave module instance to send data to.
                        - ringID (int, optional): Identifier for the Earthworm ring. Defaults to 0.
                        - scaling (float, optional): Scaling factor for the data. Defaults to 1e6.

                    Returns:
                        - None
                """
                if scaling > 1.0:
                    traceData = (self.getData()[self.chIds,:]*scaling).astype(np.int32)
                    dataFormat = 'i4'
                else:
                    # Not currently tested
                    traceData = self.getData()[self.chIds,:].astype(np.float32)
                    dataFormat = 'f4'
                npts = traceData.shape[1]
                timeStamps = self.getTimeStamps()
                timestamp_init = timeStamps[0].timestamp()
                for idx in range(len(self.chIds)):
                    wave = {
                        'station': self.stats[idx].station, 
                        'network': self.stats[idx].network, 
                        'channel': self.stats[idx].channel, 
                        'location': '--', 
                        'nsamp': npts, 
                        'samprate': fs, 
                        'startt': timestamp_init,
                        'endt': timestamp_init+(npts-1)/fs,
                        'datatype': dataFormat,
                        'data': traceData[idx,:]
                    }
                    waveMod.put_wave(ringID, wave)
                return
            
    def append(self, x, timestamps=None):
        """append an element at the end of the buffer"""
        self.data.append(np.expand_dims(x[self.good_ch], axis=1))
        if timestamps is not None:
            self.timeStamps.append(timestamps)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def getData(self):
        """ Return array of elements from the oldest to the newest. """
        return np.concatenate(self.data, axis=1)
    
    def getTimeStamps(self):
        """ Return array of related time stamps """
        return np.array(self.timeStamps)
    
    def setObspyTraceHeader(self, inventory=None):
        """Method to set channel header from inventory"""
        if inventory is None:
            self.stats = None
            self.channels_info = None
            return
        station_dict = inventory.get_contents()
        self.channels_info = station_dict['channels']
        network_codes = [stat.split(".")[0] for stat in self.channels_info]
        station_codes = [stat.split(".")[1] for stat in self.channels_info]
        channel_codes = [stat.split(".")[3] for stat in self.channels_info]
        self.chIds = [int(stat.split(" ")[-1][:-1].split("/")[1]) for stat in station_dict['stations']]
        # Creating stats for traces
        self.stats = []
        self.statNames = [] # Necessary for streaming traveltime picks
        for idx in range(len(self.chIds)):
            stat = Stats()
            stat.network = network_codes[idx]
            stat.station = station_codes[idx]
            stat.channel = channel_codes[idx]
            self.stats.append(stat)
            self.statNames.append("%s.%s.%s.--"%(station_codes[idx],channel_codes[idx],network_codes[idx]))
        self.statNames = np.array(self.statNames)
        return
    
    def writeObsPyTraces(self, fs, datapath, scaling=1e6):
        """Method to write Obspy traces"""
        if self.channels_info is None:
            raise ValueError("Call method setObspyTraceHeader to set trace info before using writeObsPyTraces")
        # Getting channel data to write and timestamps
        traceData = (self.getData()[self.chIds,:]*scaling).astype(np.int32)
        timeStamps = self.getTimeStamps()
        # Loop for writing traces
        for idx in range(len(self.chIds)):
            self.stats[idx].sampling_rate = fs
            self.stats[idx].npts = traceData.shape[1]
            self.stats[idx].starttime = UTCDateTime(timeStamps[0])
            self.stats[idx].delta = 1.0/fs
            tr = Trace(traceData[idx,:], header=self.stats[idx])
            file_path = "%s/%s_%s.mseed"%(datapath,self.channels_info[idx], timeStamps[0].strftime(time_format))
            tr.write(file_path, format="MSEED", reclen=512, encoding='STEIM2')
            os.chmod(file_path, 0o644)
        return
    
    def send2ew(self, fs, waveMod, ringID=0, scaling=1e6):
        """Function to send buffered data to Earthworm wavering"""
        if scaling > 1.0:
            traceData = (self.getData()[self.chIds,:]*scaling).astype(np.int32)
            dataFormat = 'i4'
        else:
            # Not currently tested
            traceData = self.getData()[self.chIds,:].astype(np.float32)
            dataFormat = 'f4'
        npts = traceData.shape[1]
        timeStamps = self.getTimeStamps()
        # Ensure the timestamp is in UTC
        timestamp_init = timeStamps[0].astimezone(timezone.utc).timestamp()
        print(timeStamps, timestamp_init)
        for idx in range(len(self.chIds)):
            wave = {
                'station': self.stats[idx].station, 
                'network': self.stats[idx].network, 
                'channel': self.stats[idx].channel, 
                'location': '--', 
                'nsamp': npts, 
                'samprate': fs, 
                'startt': timestamp_init,
                'endt': timestamp_init+(npts-1)/fs,
                'datatype': dataFormat,
                'data': traceData[idx,:]
            }
            waveMod.put_wave(ringID, wave)
        return


############################################################################################################
# PhaseNet-DAS realtime functions and utilities
############################################################################################################

# Function to run the event loop in a separate thread
def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Function to stop correctly thread at the exit
def close_picking_thread(thread, loop):
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    return

def start_picking_thread(device="cuda"):
    """Function to start picking thread and loading ML model on proper device"""
    # Create an event loop for the separate thread
    loop = asyncio.new_event_loop()

    # Start the event loop in a new thread
    thread = threading.Thread(target=start_event_loop, args=(loop,))
    thread.start()

    # Registering the closure of the loop once the main program has ended
    atexit.register(lambda: close_picking_thread(thread, loop))

    # Loading ML model on device
    print("Loading ML model...", end=" ", flush=True)
    
    # Make sure `DAS_ML.preload_model_async` is a coroutine
    task1 = asyncio.run_coroutine_threadsafe(DAS_ML.preload_model_async(device=device), loop)
    
    # Wait for task1 to complete
    try:
        task1.result()  # This will block until task1 completes
        print("DONE", flush=True)
    except Exception as e:
        print(f"Failed to load ML model: {e}", flush=True)
        # Ensure the thread and loop are closed if there's an error
        close_picking_thread(thread, loop)
        raise

    return loop

async def real_time_picking_async(DASdata, dt, timeStamps, minbuf=2.0, maxbuf=68.0):
    """Function performing real-time"""
    time_format_picking = "%Y-%m-%dT%H:%M:%S.%f+00:00" 
    fs = 1.0/dt
    first_timestamp = dateutil.parser.parse(timeStamps[0].strftime(time_format_picking))
    TT_picks = DAS_ML.phasenet_das(DASdata, first_timestamp.strftime(time_format_picking), 0, dt)
    if len(TT_picks) == 0:
        TT_picks = None
    else:
        TT_picks["station_id"] = TT_picks["station_id"].apply(lambda x: int(x))
        TT_picks['phase_time'] = pd.to_datetime(TT_picks['phase_time'])
        TT_picks["phase_time_seconds"] = TT_picks["phase_time"].apply(lambda x: (x - first_timestamp).total_seconds())
        TT_picks = TT_picks[(TT_picks['phase_time_seconds'] >= minbuf) & (TT_picks['phase_time_seconds'] <= maxbuf)]
        if len(TT_picks) == 0:
            return None
        TT_picks["begin_time"] = first_timestamp
        TT_picks["peak Strain rate [nm/m/s]"] = DAS_ML.extract_peak_amp(DASdata, TT_picks["phase_time_seconds"].to_numpy(), 
                                  fs , 0.0, 0.0, 25.0, chIDs=TT_picks["station_id"].to_numpy())
    return TT_picks

Pickcounter = 0
def merge_stream_picks(TT_picksBuf, TT_picksNew, delta_t_thres=2.0, maxBuf=3600.0, pickRing=None, streamCh=None, chCodes=None, qualityThresHold=[1.0,0.97,0.9,0.0], quality_values=[0,1,2,3]):
    """Function to buffer traveltime picks and stream them using a pyEarthworm pickring"""
    global Pickcounter  # Declare Pickcounter as global inside the function
    # Remove any pick with picking time greater than maxBuf
    curTimeUTC = datetime.now(timezone.utc)
    if TT_picksBuf is None and TT_picksNew is None:
        return None
    
    # Check if any picks is more than maxBuf old
    if TT_picksBuf is not None:
        TT_picksBuf["phase_time_seconds"] = TT_picksBuf["phase_time"].apply(lambda x: (curTimeUTC - x).total_seconds())
        TT_picksBuf = TT_picksBuf[TT_picksBuf['phase_time_seconds'] < maxBuf]
        if len(TT_picksBuf) == 0:
            return None

    # Checking if same time pick is present within picking buffer
    if TT_picksBuf is not None and TT_picksNew is not None:
        TT_picks = pd.concat([TT_picksBuf, TT_picksNew])
    elif TT_picksBuf is None and TT_picksNew is not None:
        TT_picks = TT_picksNew
    else:
        return TT_picksBuf

    # Sort by 'station_id', 'phase_type' and 'phase_time'
    TT_picks = TT_picks.sort_values(by=['station_id', 'phase_type', 'phase_time'])

    # Calculate the time differences within each group of 'station_id' and 'phase_type'
    TT_picks['time_diff'] = TT_picks.groupby(['station_id', 'phase_type'])['phase_time'].diff().dt.total_seconds().abs()

    # Keep only the rows where the time difference is greater than 2 seconds or is NaN (first element in the group)
    TT_picks = TT_picks[(TT_picks['time_diff'] > delta_t_thres) | (TT_picks['time_diff'].isna())]

    # Streaming new picks if available and pickRing was provided
    TT_picksNew = TT_picks[TT_picks['time_diff'].isna()]

    if len(TT_picksNew) > 0:
        # Assigning given quality to picks 
        # Convert the 'phase_score' column to numeric, forcing errors to NaN
        TT_picksNew['phase_score'] = pd.to_numeric(TT_picksNew['phase_score'], errors='coerce')
        conditions = [
            (TT_picksNew['phase_score'] >= qualityThresHold[0]),
            (TT_picksNew['phase_score'] >= qualityThresHold[1]) & (TT_picksNew['phase_score'] < qualityThresHold[0]),
            (TT_picksNew['phase_score'] >= qualityThresHold[2]) & (TT_picksNew['phase_score'] < qualityThresHold[1]),
            (TT_picksNew['phase_score'] < qualityThresHold[2])
        ]
        TT_picksNew['Q'] = np.select(conditions, quality_values)

        # Pick ring provided and streamed channel list provided?
        if pickRing is not None and streamCh is not None:
            # Checking if streamed channels have any new pick
            TT_picksNew = TT_picksNew[TT_picksNew["station_id"].isin(streamCh)]
            if len(TT_picksNew) > 0:
                # Adding new picks to pickRing
                for _, pick in TT_picksNew.iterrows():
                    chidx = np.where(streamCh == pick["station_id"])[0]
                    picktime = pick["phase_time"].strftime("%Y-%m-%dT%H%M%S.%f+00:00")[:-6].replace("-", "").replace("T","").replace(":", "")
                    pickString = "8 99 4 %s %s ?%s %s 0 0 0"%(Pickcounter, chCodes[chidx], pick['Q'], picktime)
                    pickRing.put_msg(1, 8, pickString)
                    Pickcounter += 1
        else:
            print("Cannot stream picks through pick ring without a running pickRing and a provided streamCh")
    # Drop the 'time_diff' column as it's no longer needed
    TT_picks = TT_picks.drop(columns=['time_diff'])
    return TT_picks