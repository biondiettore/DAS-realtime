#!/usr/bin/env python3
# Main script for real-time DAS data processing and streaming

import socket
import argparse
import numpy as np
from DASPacket_Mod import *
from DAS_RealTimeMod import RingBuffer, real_time_picking_async, start_picking_thread, merge_stream_picks, time_format
import time
from datetime import timedelta
from datetime import datetime, timezone
from obspy.core.inventory import inventory
import asyncio
import PyEW

time_format = "%Y-%m-%dT%H%M%SZ"

def doWork(inp_socket, args, waveRing=None, pickRing=None, loop=None):
    try:

        workInterval = args.workInterval
        strainRate = args.strainRate
        ringbuff_size = args.ringbuffer
        filelength = args.filelength
        filepath = args.filepath
        pickOutInt = args.pickOutputInterval
        if args.xmlmeta is None and filelength > 0.0:
            raise ValueError("User must provide XML meta information")
        if args.xmlmeta is not None:
            inventor = inventory.read_inventory(args.xmlmeta)
        else:
            inventor = None
        if workInterval >= ringbuff_size:
            raise ValueError("workInterval (%s) must be smaller than ringbuffer size (%s)"%(workInterval,ringbuff_size))
        taskPicking = None # Picking task to check if previous task is done
        minimumPhaseNetTime = 30.0
        TT_picksBuf = None

        ii = 0 # packet counter
        packet = getNextPacket(inp_socket)
        fs = getFs(packet)
        deltaStrainRate = timedelta(seconds=float(0.5/fs))
        nch = getNumChannel(packet)
        pickingChannel = np.arange(nch)
        if args.pickingChannel is not None:
            pickingChannel = np.loadtxt(args.pickingChannel, delimiter=',', dtype=int)
        # Picking output folder
        pickOutput = args.pickOutput
        startPicking = time.time()

        decFact = getDecimation(packet)
        ringbuff = RingBuffer(int(ringbuff_size*fs), pickingChannel)
        ringbuff.setObspyTraceHeader(inventor)
        streamCh = None
        chCodes = None
        if ringbuff.channels_info is not None:
            streamCh = np.array(ringbuff.chIds)
            chCodes = ringbuff.statNames   

        # Factors to convert phase to strain
        GaugeL = getGaugeLengthProc(packet) # [m]
        nFiber = 1.4682
        lamdLaser = 1550.0
        eta = 0.78 # photo-elastic scaling factor for longitudinal strain in isotropic material
        factor = 4.0*np.pi*eta*nFiber*GaugeL/lamdLaser
        # Conversion factor from raw to delta phase
        radconv = 1.0
        phaseLSB = getPhaseLSB(packet)
        scaleFactor = 2*np.pi/(2**phaseLSB)
        conv_factor = 1.0/factor/radconv*scaleFactor

        OldtimeSample = getPayloadRad(packet)*conv_factor
        if strainRate:
            packet = getNextPacket(inp_socket)
            currtimeSample = getPayloadRad(packet)*conv_factor
            ringbuff.append(currtimeSample-OldtimeSample, timestamps=getPacketTimestamp(packet)-deltaStrainRate)
            OldtimeSample = currtimeSample
        else:
            ringbuff.append(OldtimeSample, timestamps=getPacketTimestamp(packet))
        ii += 1
        oldSample = getSampleCount(packet)
        while True:
            packet = getNextPacket(inp_socket)
            if packet == b'': break

            # Checking if samples have been skipped
            currSample =  getSampleCount(packet)
            nSamples = currSample - oldSample
            oldSample = currSample

            if nSamples != decFact and ii > 0:
                print("Skipped samples", flush=True)
                print(f'Packet timestamp and number of samples: {getPacketTimestamp(packet)}, {nSamples}', flush=True)
                if filelength > 0.0 and filepath is not None:
                    print("Writing file at %s"%ringbuff.getTimeStamps()[-1].strftime(time_format), flush=True)
                    # Writing mseed traces
                    ringbuff.writeObsPyTraces(fs, filepath)
                time.sleep(0.1)
                return
        
            # Filling the buffer
            currtimeSample = getPayloadRad(packet)*conv_factor
            if strainRate:
                ringbuff.append(currtimeSample-OldtimeSample, timestamps=getPacketTimestamp(packet)-deltaStrainRate)
                OldtimeSample = currtimeSample
            else:
                ringbuff.append(currtimeSample, timestamps=getPacketTimestamp(packet))
            ii += 1

            # Do some processing every workInterval
            if workInterval > 0.0  and ii % int(workInterval*fs) == 0 and ii > 0:
                # print(f'Packet timestamp and shape of ringbuffer: {ringbuff.getTimeStamps()[-1]}, {ringbuff.getData().shape}', flush=True)
                # Picking has been requested if loop is not None
                if loop is not None:
                    # Performing some checks before submitting picking task
                    if taskPicking is not None:
                        # Previous picking done?
                        if (not taskPicking.done()):
                            continue
                        else:
                            TT_picksNew = taskPicking.result()
                            # Picking streaming task
                            TT_picksBuf = merge_stream_picks(TT_picksBuf, TT_picksNew, delta_t_thres=2.0, maxBuf=pickOutInt, pickRing=pickRing, streamCh=streamCh, chCodes=chCodes)
                            # Checking if writing of picking data base was requested
                            if time.time()-startPicking >=pickOutInt and pickOutput is not None:
                                if TT_picksBuf is not None:
                                    filename = pickOutput+"/%s.csv"%datetime.fromtimestamp(startPicking).astimezone(timezone.utc).strftime(time_format)
                                    print("Writing picking file %s"%filename, flush=True)
                                    # Dropping duplicated picks before writting file
                                    TT_picksBuf = TT_picksBuf.drop_duplicates(subset=['station_id', 'phase_time'])
                                    TT_picksBuf.to_csv(filename, index=False)
                                startPicking = time.time()
                    if ringbuff.getData().shape[1] < minimumPhaseNetTime*fs:
                        # Data buffer not large enough to perform picking; waiting for more data to come in
                        continue 
                    taskPicking = real_time_picking_async(ringbuff.getData().copy(), 1.0/fs, ringbuff.getTimeStamps())
                    taskPicking = asyncio.run_coroutine_threadsafe(taskPicking, loop)
            # Change ringbuff_size to filelength once final version is implemented
            if filelength > 0.0 and ii % int(ringbuff_size*fs) == 0 and ii > 0 and filepath is not None:
                print("Writing file at %s"%ringbuff.getTimeStamps()[-1].strftime(time_format), flush=True)
                # Writing mseed traces
                ringbuff.writeObsPyTraces(fs, filepath)

            # Streaming data to the RingServer (SeedLink) using simpledali
            if waveRing is not None and ii % int(ringbuff_size*fs) == 0:
                ringbuff.send2ew(fs, waveRing)
          
    except Exception as e:
        print(e)
        pass


waveRing = None
pickRing = None
loop = None
def main():
    """Main function to perform picking and earthworm picks and data streaming"""
    # Global variables
    global waveRing, pickRing, loop # Necessary to avoid restart of pick or/and wave rings
    # Perform some checks on arguments.
    try:
        socket.inet_aton(args.host)
    except socket.error:
        error_exit("Invalid input host IP address")
    if 1 > args.port > 65535:
        error_exit("Invalid input port number")

    #Open source socket
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as inp:
            try:
                print(f'Connecting to the data stream {args.host}:{args.port}...', flush=True)
                inp.connect((args.host, args.port))
            except Exception as e:
                error_exit("Cannot connect to the data stream host: {e}")
            print('Connected', flush=True)
            # Checking if wave Ring parameters were passed
            if args.wavering[0] and waveRing is None:
                ringNumber = args.wavering[0]
                modID = args.wavering[1]
                inst_id = args.wavering[2]
                hb_freq = float(args.wavering[3])
                db_ew = bool(args.wavering[4])
                # Connecting to existing Earthworm ring
                waveRing = PyEW.EWModule(ringNumber, modID, inst_id, hb_freq, db_ew)
                # Adding wave ring to pyEw Module object
                waveRing.add_ring(ringNumber)
            if args.pickring[0] and pickRing is None:
                ringNumber = args.pickring[0]
                modID = args.pickring[1]
                inst_id = args.pickring[2]
                hb_freq = float(args.pickring[3])
                db_ew = bool(args.pickring[4])
                # Connecting to existing Earthworm ring
                pickRing = PyEW.EWModule(ringNumber, modID, inst_id, hb_freq, db_ew)
                # Adding wave ring to pyEw Module object
                pickRing.add_ring(ringNumber)
            # Checking if picking wave requested
            if args.picking and loop is None:
                loop = start_picking_thread(args.device)
            # Starting processing 
            doWork(inp, args, loop=loop, waveRing=waveRing, pickRing=pickRing)
            print('Got disconnected from the data stream', flush=True)
            inp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main program to process real-time DAS data streams.\n")
    parser.add_argument('--host', metavar='HOST', required=True, help='a hostname of the input data stream')
    parser.add_argument('--port', metavar='PORT', type=int, required=True, help='a port of the input data stream')
    parser.add_argument('--strainRate', '-strnRt', metavar='strainRate', type=int, default=1, help='Save strain rate or strain. Default 1')
    parser.add_argument('--workInterval', '-wrkint', metavar='workInterval', type=float, default=1.0, help='Work interval for data processing. Default 1.0 [s]')
    parser.add_argument('--picking', '-pick', metavar='picking', type=int, default=0, help='Flag to run picking using PhaseNet-DAS (Zhu et al., 2023)')
    parser.add_argument('--pickingChannel', '-pch', metavar='pickingChannel', type=str, default=None, help='File name containing the indices of the channels to consider during the picking process (Comma-separated integer values).')
    parser.add_argument('--pickOutput', '-pOut', metavar='pickOutput', type=str, default=None, help='Folder where picking database are written. Currently hourly output.')
    parser.add_argument('--pickOutputInterval', '-pOutInt', metavar='pickOutInt', type=float, default=3600.0, help='Interval to write picks into .csv file. Default 3600.0 [s]')
    parser.add_argument('--pickring', '-pring', metavar='pickring', type=int, nargs=5, default=[None,8,141,30,0], help='PyEarthworm module parameters for pick ring (ring number,module ID, INST_ID,HeartBeats frequency [s], Debug Flag); see parameters for PyEW.EWModule')
    parser.add_argument('--device', '-dev', metavar='device', type=str, default="cuda", help='Device on which to run PhaseNet-DAS; picking must be 1 to work. To use different GPU card use cuda:1 to select GPU card ID 1')
    parser.add_argument('--ringbuffer', '-Rbfsz', metavar='ringbuffer', type=float, default=60.0, help='Ring buffer size in seconds to be stored for processing. Default 60.0 [s]')
    parser.add_argument('--filelength', '-flng', metavar='filelength', type=float, default=0.0, help='Interval to be writing data to disk in seconds. Default 0.0 [s], meaning no file writing')
    parser.add_argument('--filepath', '-flpt', metavar='filepath', type=str, default=None, help='Path for writing the mseed files')
    parser.add_argument('--xmlmeta', '-xml', metavar='xmlmeta', type=str, default=None, help='Path to the XML metadata channel info')
    parser.add_argument('--wavering', '-wring', metavar='wavering', type=int, nargs=5, default=[None,8,141,30,0], help='PyEarthworm module parameters for wave ring (ring number,module ID, INST_ID,HeartBeats frequency [s], Debug Flag); see parameters for PyEW.EWModule')
    parser.add_argument('--debug', '-dbg', metavar='debug', type=int, default=0, help='Debug flag for asyncio module')
    args = parser.parse_args()
    # Running main processing streaming function
    main()