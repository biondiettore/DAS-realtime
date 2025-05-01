import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import json, struct

def error_exit(message, code=1):
    sys.stderr.write("Error: {0}\n".format(str(message)))
    sys.exit(code)

class StreamReader:
    """Template class for any Stream data reader"""

    def getNextPacket(self):
        raise NotImplementedError("getNextPacket must be overridden")
        
    def getPacketTimestamp(self, header):
        """Returns a timestamp from the header."""
        raise NotImplementedError("getPacketTimestamp must be overridden")

    def getNumTimeSamples(self, packet):
        raise NotImplementedError("getNumTimeSamples must be overridden")

    def getSampleCount(self, packet):
        raise NotImplementedError("getSampleCount must be overridden")

    def getNumChannel(self, packet):
        raise NotImplementedError("getNumChannel must be overridden")

    def getGaugeLengthProc(self, packet):
        raise NotImplementedError("getGaugeLengthProc must be overridden")

    def getHeader(self, packet):
        raise NotImplementedError("getHeader must be overridden")

    def getPayload(self, packet):
        raise NotImplementedError("getPayload must be overridden")

    def getPayloadRad(self, packet):
        raise NotImplementedError("getPayloadRad must be overridden")

    def getDecfactorORsamplesPerPacket(self, packet):
        raise NotImplementedError("getDecfactorORsamplesPerPacket must be overridden")

    def getFs(self, packet):
        raise NotImplementedError("getFs must be overridden")

    def getConversionFactor(self, packet):
        raise NotImplementedError("getConversionFactor must be overridden")
    

class ASN_StreamReader(StreamReader):
    """Streamer Reader for ASN data"""
    def __init__(self, socket):
        self.socket = socket
        self.header = None
        self.rois = []
        self.times = []
        self.data = []
        self.SampleCount = 0
        self.unpackFormat = ""
        self.readHeader(self.socket.recv().decode('utf-8'))

    def readHeader(self, text):
        header = json.loads(text)
        unpackFormat = f"{header['nChannels'] * header['nPackagesPerMessage']}{'f' if header['dataType']=='float' else 'h'}"
        rois = [] # List with roi slices. 
        channel = 0
        for roi in header['roiTable']:
            channels = (roi['roiEnd'] -  roi['roiStart']) //  roi['roiDec'] +1
            rois.append(slice(channel, channel+channels, 1))
            channel += channels
        self.header = header
        self.rois = rois
        self.unpackFormat = unpackFormat
        return 

    def getHeader(self, packet):
        return self.header
       
    def getROIs(self, packet):
        return self.rois

    def timeString(self, timestamp):
        return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%fZ')

    def getNextPacket(self):
        """Function to obtain next packet; for ASN unpacking everything directly"""
        packet = self.socket.recv()
        if len(packet) != self.header['bytesPerPackage']*self.header['nPackagesPerMessage'] + 8:
            print('Parameters Changed. Updated header, ROIs and format; restarting data stream')
            return b''
        # Getting timestamp
        timestamp = struct.unpack('Q' , packet[:8])[0]*1e-9 # Nanoseconds to seconds
        time_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        dt =  self.header["dt"]
        # Reading the data raw
        self.data = np.array(struct.unpack(self.unpackFormat, packet[8:])).reshape((self.header['nPackagesPerMessage'], self.header['nChannels']))
        # Updating internal sample count
        self.SampleCount += self.data.shape[0]
        self.times = [time_utc + timedelta(seconds=(it*dt)) for it in np.arange(self.data.shape[0])]
        return packet
        
    def getPacketTimestamp(self, header):
        """Returns the timestamps of the data points"""
        return self.times

    def getNumTimeSamples(self, packet):
        return self.data.shape[0]

    def getSampleCount(self, packet):
        return self.SampleCount

    def getNumChannel(self, packet):
        return self.header['nChannels']

    def getGaugeLengthProc(self, packet):
        return self.header['gaugeLength']

    def getPayload(self, packet):
        raise NotImplementedError("getPayload must be overridden")

    def getPayloadRad(self, packet):
        return self.data

    def getDecfactorORsamplesPerPacket(self, packet):
        return self.header["nPackagesPerMessage"]

    def getFs(self, packet):
        return 1.0/self.header["dt"]

    def getConversionFactor(self, packet):
        return 1.0/(self.header["sensitivities"][0]["factor"])



class OptaSenseStreamReader(StreamReader):
    """Streamer Reader for OptaSense data"""
    def __init__(self, socket):
        self.buffer = b''
        self.socket = socket
        raise NotImplementedError("OptaSenseStreamReader not available as open source. Contact Luna Innovations to request the code.")

    def getNextPacket(self):
        raise NotImplementedError("getNextPacket must be overridden")
        
    def getPacketTimestamp(self, header):
        """Returns a timestamp from the header."""
        raise NotImplementedError("getPacketTimestamp must be overridden")

    def getNumTimeSamples(self, packet):
        raise NotImplementedError("getNumTimeSamples must be overridden")

    def getSampleCount(self, packet):
        raise NotImplementedError("getSampleCount must be overridden")

    def getNumChannel(self, packet):
        raise NotImplementedError("getNumChannel must be overridden")

    def getGaugeLengthProc(self, packet):
        raise NotImplementedError("getGaugeLengthProc must be overridden")

    def getHeader(self, packet):
        raise NotImplementedError("getHeader must be overridden")

    def getPayload(self, packet):
        raise NotImplementedError("getPayload must be overridden")

    def getPayloadRad(self, packet):
        raise NotImplementedError("getPayloadRad must be overridden")

    def getDecfactorORsamplesPerPacket(self, packet):
        raise NotImplementedError("getDecfactorORsamplesPerPacket must be overridden")

    def getFs(self, packet):
        raise NotImplementedError("getFs must be overridden")

    def getConversionFactor(self, packet):
        raise NotImplementedError("getConversionFactor must be overridden")