# Module containing real-time processing classes and functions
import numpy as np
from obspy.core.trace import Trace 
from obspy.core.trace import Stats
from obspy.core.utcdatetime import UTCDateTime
import os, sys

# SeedLink necessary 
import asyncio
from  simplemseed import MiniseedHeader, MiniseedRecord, MSeed3Header, MSeed3Record, encodeSteim2, encodeSteim2FrameBlock, seedcodec

# Necessary to run PhaseNet-DAS in real time
import threading
import dateutil.parser
import atexit
import pandas as pd
# Adding pyDAS location
DAS_util_path = "/home/ebiondi/packages/DAS-utilities/"
pyDAS_path = DAS_util_path+"build/" # Substitute this path with yours
try:
    os.environ['LD_LIBRARY_PATH'] += ":" + pyDAS_path
except:
    os.environ['LD_LIBRARY_PATH'] = pyDAS_path
sys.path.insert(0,pyDAS_path)
sys.path.insert(0,DAS_util_path+'python/') # Substitute this path with yours
import DASutils
import DAS_ML


############################################################################################################
# DAS RingBuffer
############################################################################################################

time_format = "%Y-%m-%dT%H%M%SZ"

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self, buff_size):
        """
        Input:
        buff_size [int]: maximum number of time samples in the ring buffer  
        """
        if buff_size < 0:
            raise ValueError("buff_size must be a positive integer")
        self.max = buff_size
        self.data = []
        self.channels_info = None
        self.timeStamps = [] # Rolling buffer of timestamp axis

    class __Full:
            """ class that implements a full buffer """
            def append(self, x, timestamps=None):
                """ Append an element overwriting the oldest one. """
                self.data[self.cur] = np.expand_dims(x, axis=1)
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
                timestamp_init = timeStamps[0].timestamp()
                for idx in range(len(self.chIds)):
                    wave = {
                        'station': self.stats[idx].station, 
                        'network': self.stats[idx].network, 
                        'channel': self.stats[idx].channel, 
                        'location': '', 
                        'nsamp': npts, 
                        'samprate': fs, 
                        'startt': timestamp_init,
                        'endt': timestamp_init+(npts-1)/fs,
                        'datatype': dataFormat,
                        'data': traceData[idx,:]
                    }
                    waveMod.put_wave(ringID, wave)
                return
            
            async def sendMSEED3dali(self, fs, dali, scaling=1e6, encoding="", blocks=63, mseed="MSEED2"):
                """Function to send packet through Ringserver"""
                # Getting channel data to write and timestamps
                traceData = (self.getData()[self.chIds,:]*scaling).astype(np.int32)
                timeStamps = self.getTimeStamps()
                if mseed == "MSEED3":
                    for idx in range(len(self.chIds)):
                        network = self.stats[idx].network
                        station = self.stats[idx].station
                        location = ""
                        channel = self.stats[idx].channel
                        # numsamples = traceData[idx,:].shape[0]
                        sampleRate = fs
                        starttime = timeStamps[0]
                        header = MSeed3Header()
                        header.starttime = starttime
                        header.sampleRatePeriod = sampleRate
                        header.network = network
                        header.station = " "+station
                        header.location = location
                        header.channel = channel
                        identifier = "FDSN:%s%s_DAS_%s_H_HHS_1"%(network,station,channel)
                        if encoding == "STEIM2":
                            header.encoding = seedcodec.STEIM2
                            data = traceData[idx,:]
                            while len(data) > 0:
                                frameBlock = encodeSteim2FrameBlock(data, blocks)
                                encoded = frameBlock.pack()
                                ms3record = MSeed3Record(header, identifier, encoded)
                                sendResult = await dali.writeMSeed3(ms3record)
                                data = data[frameBlock.numSamples:]
                        else:
                            Data = traceData[idx,:]
                            ms3record = MSeed3Record(header, identifier, Data)
                            sendResult = await dali.writeMSeed3(ms3record)
                elif mseed == "MSEED2":
                        for idx in range(len(self.chIds)):
                            network = self.stats[idx].network
                            station = self.stats[idx].station
                            location = ""
                            channel = self.stats[idx].channel
                            # numsamples = traceData[idx,:].shape[0]
                            sampleRate = fs
                            starttime = timeStamps[0]
                            if encoding == "STEIM2":
                                msh = MiniseedHeader(network, station, location, channel, starttime, data.shape[0], sampleRate)
                                msh.encoding = seedcodec.STEIM2
                                data = traceData[idx,:]
                                encoded = encodeSteim2(data)
                                msr = MiniseedRecord(msh, data=None, encodedData=encoded)
                                sendResult = await dali.writeMSeed(msr)
                                # while len(data) > 0:
                                #     frameBlock = encodeSteim2FrameBlock(data, blocks)
                                #     encoded = frameBlock.pack()
                                #     msh = MiniseedHeader(network, station, location, channel, starttime, data.shape[0], sampleRate)
                                #     msr = MiniseedRecord(msh, encoded)
                                #     sendResult = await dali.writeMSeed(msr)
                                #     data = data[frameBlock.numSamples:]
                            else:
                                Data = traceData[idx,:]
                                msh = MiniseedHeader(network, station, location, channel, starttime, Data.shape[0], sampleRate)
                                msr = MiniseedRecord(msh, Data)
                                sendResult = await dali.writeMSeed(msr)
                else:
                    raise ValueError("Provided unknown mseed format: %s"%mseed)
                return

    def append(self, x, timestamps=None):
        """append an element at the end of the buffer"""
        self.data.append(np.expand_dims(x, axis=1))
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
        for idx in range(len(self.chIds)):
            stat = Stats()
            stat.network = network_codes[idx]
            stat.station = station_codes[idx]
            stat.channel = channel_codes[idx]
            self.stats.append(stat)
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
        timestamp_init = timeStamps[0].timestamp()
        for idx in range(len(self.chIds)):
            wave = {
                'station': self.stats[idx].station, 
                'network': self.stats[idx].network, 
                'channel': self.stats[idx].channel, 
                'location': '', 
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

async def real_time_picking_async(DASdata, dt, timeStamps):
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
        # TT_picks = TT_picks[(TT_picks['phase_time_seconds'] >= 15) & (TT_picks['phase_time_seconds'] <= 70)]
        TT_picks["begin_time"] = first_timestamp
        TT_picks["peak Strain rate [nm/m/s]"] = DASutils.extract_peak_amp(DASdata, TT_picks["phase_time_seconds"].to_numpy(), 
                                  fs , 0.0, 0.0, 25.0, chIDs=TT_picks["station_id"].to_numpy())
    return TT_picks

    
