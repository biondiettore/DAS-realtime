#!/usr/bin/env python3
# Main script for real-time DAS data processing and streaming

import socket
import argparse
import numpy as np
from CBT_OptaSenseMod import *
from DAS_RealTimeMod import RingBuffer, real_time_picking_async, start_picking_thread
import time
from datetime import timedelta
from obspy.core.inventory import inventory
import asyncio
import os
import logging, PyEW

# SeedLink necessary
import simpledali
# DALI parameters
programname = "DASstreaming"
processid = 0
architecture = "python"
username =  os.getlogin()  
processid = 1

time_format = "%Y-%m-%dT%H%M%SZ"

async def doWork_async(inp_socket, args, dali=None, loop=None):
    try:

        workInterval = args.workInterval
        strainRate = args.strainRate
        ringbuff_size = args.ringbuffer
        filelength = args.filelength
        filepath = args.filepath
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

        ii = 0 # packet counter
        packet = getNextPacket(inp_socket)
        fs = getFs(packet)
        deltaStrainRate = timedelta(seconds=float(0.5/fs))
        nch = getNumChannel(packet)
        decFact = getDecimation(packet)
        ringbuff = RingBuffer(int(ringbuff_size*fs))
        ringbuff.setObspyTraceHeader(inventor)
        
        # Factors to convert phase to strain
        GaugeL = getGaugeLengthProc(packet) # [m]
        ChSamp = 10.209524 # [m]
        nFiber = 1.4682
        lamdLaser = 1550.0
        eta = 0.78 # photo-elastic scaling factor for longitudinal strain in isotropic material
        factor = 4.0*np.pi*eta*nFiber*GaugeL/lamdLaser
        # Conversion factor from raw to delta phase
        radconv = 1.0#10430.378850470453
        # DAS_data = DAS_data/factor/radconv*1e6
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
        aveLoopTime = 0.0
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
                print(f'Packet timestamp and shape of ringbuffer: {ringbuff.getTimeStamps()[-1]}, {ringbuff.getData().shape}', flush=True)
                # Picking has been requested if loop is not None
                if loop is not None:
                    # Performing some checks before submitting picking task
                    if taskPicking is not None:
                        # Previous picking done?
                        if (not taskPicking.done()):
                            continue
                        else:
                            print(taskPicking.result())
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
            if dali is not None and ii % int(ringbuff_size*fs) == 0:
                await ringbuff.sendMSEED3dali(fs, dali)

                
    except Exception as e:
        print(e)
        pass

def doWork(inp_socket, args, waveRing=None, loop=None):
    try:

        workInterval = args.workInterval
        strainRate = args.strainRate
        ringbuff_size = args.ringbuffer
        filelength = args.filelength
        filepath = args.filepath
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

        ii = 0 # packet counter
        packet = getNextPacket(inp_socket)
        fs = getFs(packet)
        deltaStrainRate = timedelta(seconds=float(0.5/fs))
        nch = getNumChannel(packet)
        decFact = getDecimation(packet)
        ringbuff = RingBuffer(int(ringbuff_size*fs))
        ringbuff.setObspyTraceHeader(inventor)
        
        # Factors to convert phase to strain
        GaugeL = getGaugeLengthProc(packet) # [m]
        ChSamp = 10.209524 # [m]
        nFiber = 1.4682
        lamdLaser = 1550.0
        eta = 0.78 # photo-elastic scaling factor for longitudinal strain in isotropic material
        factor = 4.0*np.pi*eta*nFiber*GaugeL/lamdLaser
        # Conversion factor from raw to delta phase
        radconv = 1.0#10430.378850470453
        # DAS_data = DAS_data/factor/radconv*1e6
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
        aveLoopTime = 0.0
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
                print(f'Packet timestamp and shape of ringbuffer: {ringbuff.getTimeStamps()[-1]}, {ringbuff.getData().shape}', flush=True)
                # Picking has been requested if loop is not None
                if loop is not None:
                    # Performing some checks before submitting picking task
                    if taskPicking is not None:
                        # Previous picking done?
                        if (not taskPicking.done()):
                            continue
                        else:
                            print(taskPicking.result())
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


async def main_async():
    """Main function for DALI functionality; cannot work with PyEW"""
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
            loop = None
            if args.daliport is None:
                if args.picking:
                    loop = start_picking_thread(args.device)
                await doWork_async(inp, args, loop=loop)
            elif args.daliport:
                async with simpledali.SocketDataLink(args.dalihost, args.daliport, verbose=args.debug) as dali:
                    serverId = await dali.id(programname, username, processid, architecture)
                    print(f"Response Ring Server: {serverId}")
                    await doWork_async(inp, args, dali=dali)

            print('Got disconnected from the data stream', flush=True)
            inp.close()

def main():
    """Main function to perform picking and earthworm picks and data streaming"""
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
            waveRing = None
            if args.wavering[0]:
                ringNumber = args.wavering[0]
                modID = args.wavering[1]
                inst_id = args.wavering[2]
                hb_freq = float(args.wavering[3])
                db_ew = bool(args.wavering[3])
                # Connecting to existing Earthworm ring
                waveRing = PyEW.EWModule(ringNumber, modID, inst_id, hb_freq, db_ew)
                # Adding wave ring to pyEw Module object
                waveRing.add_ring(ringNumber)
            # Checking if picking wave requested
            loop = None
            if args.picking:
                loop = start_picking_thread(args.device)
            # Starting processing 
            doWork(inp, args, loop=loop, waveRing=waveRing)
            print('Got disconnected from the data stream', flush=True)
            inp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple CBT stream reader example.\n")
    parser.add_argument('--host', metavar='HOST', required=True, help='a hostname of the input data stream')
    parser.add_argument('--port', metavar='PORT', type=int, required=True, help='a port of the input data stream')
    parser.add_argument('--strainRate', '-strnRt', metavar='strainRate', type=int, default=1, help='Save strain rate or strain. Default 1')
    parser.add_argument('--workInterval', '-wrkint', metavar='workInterval', type=float, default=1.0, help='Work interval for data processing. Default 1.0 [s]')
    parser.add_argument('--picking', '-pick', metavar='picking', type=int, default=0, help='Flag to run picking using PhaseNet-DAS (Zhu et al., 2023)')
    parser.add_argument('--device', '-dev', metavar='device', type=str, default="cuda", help='Device on which to run PhaseNet-DAS; picking must be 1 to work and daliport None')
    parser.add_argument('--ringbuffer', '-Rbfsz', metavar='ringbuffer', type=float, default=60.0, help='Ring buffer size in seconds to be stored for processing. Default 60.0 [s]')
    parser.add_argument('--filelength', '-flng', metavar='filelength', type=float, default=0.0, help='Interval to be writing data to disk in seconds. Default 0.0 [s], meaning no file writing')
    parser.add_argument('--filepath', '-flpt', metavar='filepath', type=str, default=None, help='Path for writing the mseed files')
    parser.add_argument('--xmlmeta', '-xml', metavar='xmlmeta', type=str, default=None, help='Path to the XML metadata channel info')
    parser.add_argument('--daliport', '-dlprt', metavar='daliport', type=int, default=None, help='Port to stream DAS data using simpledali')
    parser.add_argument('--dalihost', '-dlhst', metavar='dalihost', type=str, default=socket.gethostname(), help='Hostname to stream DAS data using simpledali')
    parser.add_argument('--wavering', '-wring', metavar='wavering', type=int, nargs=5, default=[None,8,141,30,0], help='PyEarthworm module parameters for wave ring (ring number,module ID, INST_ID,HeartBeats frequency [s], Debug Flag); see parameters for PyEW.EWModule')
    parser.add_argument('--debug', '-dbg', metavar='debug', type=int, default=0, help='Debug flag for asyncio module')
    args = parser.parse_args()
    if args.daliport is None:
        main()
    else:
        asyncio.run(main_async(), debug=args.debug)