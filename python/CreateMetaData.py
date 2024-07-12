#!/usr/bin/env python3
# Script to create metadata for a few specified channels

import obspy
from obspy.core.inventory import Inventory, Network, Station, Channel, Site, Response
from obspy.core.inventory.response import InstrumentSensitivity
from obspy.core.inventory.util import Equipment
import argparse
import utm
import numpy as np
import pandas as  pd

def azimuth_dipping_from_gps(xlon, ylat, zele):
    """
    get azimuth and dipping angle from gps locations: longitude, latitude, and elevation
    Args:
        xlon: longitude array in degree
        ylat: latitude array in degree
        xele: elevaton array in meter
    Returns:
        azimuth: angle zero in North direction, positive for left hand rotation w.r.t Up
        dipping: angle zero in horizontal direction, ranges from -90.0 to 90.0 degree
    """
    utm_reference = utm.from_latlon(ylat[0], xlon[0])
    utm_coords = utm.from_latlon(ylat, xlon, utm_reference[2], utm_reference[3])
    nx = len(xlon)
    azimuth = np.zeros(nx)
    dipping = np.zeros(nx)
    unit_z = np.array([0,0,1])
    r2d = 180/np.pi
    for i in range(nx):
        i1 = i if i == 0 else i-1
        i2 = i if i == nx-1 else i+1
        # unit vectors
        vec_en = np.array([utm_coords[0][i2]-utm_coords[0][i1], utm_coords[1][i2]-utm_coords[1][i1]])
        vec_enz = np.append(vec_en, zele[i2]-zele[i1])
        vec_en = vec_en/np.sqrt(np.sum(vec_en**2)+1e-10)
        vec_enz = vec_enz/np.sqrt(np.sum(vec_enz**2)+1e-10)
        # azimuth
        azi = np.arctan2(vec_en[0], vec_en[1])*r2d
        if azi < 0:
            azi += 360
        # dipping
        dip = np.arccos(np.dot(vec_enz, unit_z))*r2d
        #
        azimuth[i] = azi
        dipping[i] = dip - 90.0
    return azimuth, dipping


if __name__ == "__main__":
     # Parsing command line
    parser = argparse.ArgumentParser(description='Program to write XML channel metadata for given channels')
    parser.add_argument("das_positions", help="CSV file with das channel locations", type=str)
    parser.add_argument("--desamplingStream", "-dsmpstream", help="Desampling factor on channels during streaming", default=1, type=int)
    parser.add_argument("--desampling", "-dsmp", help="Desampling factor for selected channels", default=100, type=int)
    parser.add_argument("--shiftCh", "-shft", help="Channel shift to avoid sampling first channel", default=50, type=int)

    # Command-line arguments 
    args = parser.parse_args()
    dasfile = args.das_positions
    desamp = args.desampling
    desampStrm = args.desamplingStream
    shiftCh = args.shiftCh

    GaugeL = 102.09524 # [m]
    ChSamp = 10.209524 # [m]
    nFiber = 1.4682
    lamdLaser = 1550.0
    fs = 100 #[sample per seconds]

    # Parsing channel locations
    ch_db = pd.read_csv(dasfile)
    ch_db = ch_db[::desampStrm]
    mapped_channels = ch_db[ch_db["status"] == "good"]["channel"].astype(int).to_numpy()
    mapped_lat = ch_db[ch_db["status"] == "good"]["latitude"].astype(float).to_numpy()
    mapped_lon = ch_db[ch_db["status"] == "good"]["longitude"].astype(float).to_numpy()
    mapped_ele = ch_db[ch_db["status"] == "good"]["elevation"].astype(float).to_numpy()
    ch_idx = np.arange(len(mapped_lat))
    ch_tot = mapped_lat.shape[0]
    # Computing azimuth
    azi, dip = azimuth_dipping_from_gps(mapped_lon, mapped_lat, mapped_ele)
    selCh = mapped_channels[::desamp] # Channel select
    selCh[0] += shiftCh
    selChDesamp = np.array([np.where(mapped_channels==ch)[0][0] for ch in selCh])
    mapped_lat_sel = mapped_lat[::desamp]
    mapped_lon_sel = mapped_lon[::desamp]
    mapped_ele_sel = mapped_ele[::desamp]
    azi_sel = azi[::desamp]
    dip_sel = dip[::desamp]

    print("Creating Meta data for %s channels"%selCh.shape[0])
    print("One channel every ~%5.2f [km]"%(ChSamp*desamp*1e-3))

    # Start date of experiment 
    startDate = obspy.UTCDateTime(1926, 1, 1)

    # Instrument response
    value = 1.0
    freq = 1.0
    instr_sen = InstrumentSensitivity(value, freq, "rad", "1/s", input_units_description="Measured phase by interrogator unit", output_units_description="Strain rate")
    response = Response(instrument_sensitivity=instr_sen)

    # Equiptment information
    equipt = Equipment(type="Distributed Acoustic Sensing Unit", installation_date=startDate, manufacturer="OptaSense", model="Plexus", description="Gauge Length: %s[m]; Laser wavelength: %s [nm]"%(GaugeL,nFiber))


    # Inventory
    inv = Inventory(networks=[],source="SCSN")

    net = Network(
            code="CI",
            stations=[],
            description="Ridgecrest south 100 km array Plexus",
            start_date=startDate)
    
    for ich in range(selCh.shape[0]):

        sta_code = str(ich).zfill(2)

        sta = Station(
            # This is the station code according to the SEED standard.
            # code="DRS%s"%sta_code,
            code=" RS%s"%sta_code,
            latitude=mapped_lat_sel[ich],
            longitude=mapped_lon_sel[ich],
            elevation=mapped_ele_sel[ich],
            creation_date=obspy.UTCDateTime(2023, 1, 1),
            start_date=obspy.UTCDateTime(2023, 1, 1),
            site=Site(name="Ridgecrest DAS channel %s/%s"%(selCh[ich],selChDesamp[ich])),
            equipments=[equipt])

        cha = Channel(
            # This is the channel code according to the SEED standard.
            code="HS1",
            # This is the location code according to the SEED standard.
            location_code="",
            start_date=startDate,
            # Note that these coordinates can differ from the station coordinates.
            latitude=mapped_lat_sel[ich],
            longitude=mapped_lon_sel[ich],
            elevation=mapped_ele_sel[ich],
            depth=1.0, # Cable burried below the surface
            azimuth=azi_sel[ich],
            dip=dip_sel[ich],
            sample_rate=fs)
        
        cha.response = response
        sta.channels.append(cha)
        net.stations.append(sta)
    inv.networks.append(net)

    print(inv)
    # inv.write("Meta/DAS_RidgecrestSouth100km.xml", format="stationxml", validate=True)
    inv.write("Meta/DAS_RidgecrestSouth100kmV2.xml", format="stationxml", validate=True)
    exit(0)
