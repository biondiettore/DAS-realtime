#!/usr/bin/env python3
# Script to create metadata for a few specified channels

import obspy
from obspy.core.inventory import Inventory, Network, Station, Channel, Site, Response
from obspy.core.inventory.response import InstrumentSensitivity
from obspy.core.inventory.util import Equipment
import argparse
import utm
import numpy as np
import pandas as pd

def azimuth_dipping_from_gps(xlon, ylat, zele):
    """
    Get azimuth and dipping angle from GPS locations: longitude, latitude, and elevation
    """
    utm_reference = utm.from_latlon(ylat[0], xlon[0])
    utm_coords = utm.from_latlon(ylat, xlon, utm_reference[2], utm_reference[3])
    nx = len(xlon)
    azimuth = np.zeros(nx)
    dipping = np.zeros(nx)
    unit_z = np.array([0, 0, 1])
    r2d = 180 / np.pi
    for i in range(nx):
        i1 = i if i == 0 else i - 1
        i2 = i if i == nx - 1 else i + 1
        vec_en = np.array([utm_coords[0][i2] - utm_coords[0][i1], utm_coords[1][i2] - utm_coords[1][i1]])
        vec_enz = np.append(vec_en, zele[i2] - zele[i1])
        vec_en = vec_en / np.sqrt(np.sum(vec_en**2) + 1e-10)
        vec_enz = vec_enz / np.sqrt(np.sum(vec_enz**2) + 1e-10)
        azi = np.arctan2(vec_en[0], vec_en[1]) * r2d
        if azi < 0:
            azi += 360
        dip = np.arccos(np.dot(vec_enz, unit_z)) * r2d
        azimuth[i] = azi
        dipping[i] = dip - 90.0
    return azimuth, dipping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program to write XML channel metadata for given channels')
    parser.add_argument("das_positions", help="CSV file with DAS channel locations", type=str)
    parser.add_argument("--desamplingStream", "-dsmpstream", default=1, type=int)
    parser.add_argument("--desampling", "-dsmp", type=int)
    parser.add_argument("--shiftCh", "-shft", default=0, type=int)

    # Metadata arguments
    parser.add_argument("--GaugeL", type=float, help="Gauge length in meters")
    parser.add_argument("--ChSamp", type=float, help="Channel spacing in meters")
    parser.add_argument("--nFiber", type=float, default=1.4682, help="Fiber refractive index")
    parser.add_argument("--lamdLaser", type=float, default=1550.0, help="Laser wavelength in nm")
    parser.add_argument("--fs", type=float, default=100.0, help="Sampling frequency (Hz)")
    parser.add_argument("--manufacturer", type=str, default="", help="DAS interrogator manufacturer")
    parser.add_argument("--model", type=str, default="", help="DAS interrogator model")
    parser.add_argument("--source", type=str, default="", help="Metadata source")
    parser.add_argument("--network", type=str, default="", help="Seismic network code")
    parser.add_argument("--array_name", type=str, default="", help="Array name prefix for station codes")
    parser.add_argument("--description", type=str, default="", help="Array description")
    parser.add_argument("--output", type=str, help="Output StationXML file")

    args = parser.parse_args()

    # Assign parsed values
    dasfile = args.das_positions
    desamp = args.desampling
    desampStrm = args.desamplingStream
    shiftCh = args.shiftCh

    GaugeL = args.GaugeL
    ChSamp = args.ChSamp
    nFiber = args.nFiber
    lamdLaser = args.lamdLaser
    fs = args.fs
    manufacturer = args.manufacturer
    model = args.model
    source = args.source
    network = args.network
    array_name = args.array_name
    description = args.description
    output_file = args.output

    # Load channel locations
    ch_db = pd.read_csv(dasfile)
    ch_db = ch_db[::desampStrm]
    good_channels = ch_db[ch_db["status"] == "good"]
    mapped_channels = good_channels["channel"].astype(int).to_numpy()
    mapped_lat = good_channels["latitude"].astype(float).to_numpy()
    mapped_lon = good_channels["longitude"].astype(float).to_numpy()
    mapped_ele = good_channels["elevation"].astype(float).to_numpy()

    ch_idx = np.arange(len(mapped_lat))
    ch_tot = mapped_lat.shape[0]

    azi, dip = azimuth_dipping_from_gps(mapped_lon, mapped_lat, mapped_ele)

    selCh = mapped_channels[::desamp]
    selCh[0] += shiftCh
    selChDesamp = np.array([np.where(mapped_channels == ch)[0][0] for ch in selCh])
    mapped_lat_sel = mapped_lat[::desamp]
    mapped_lon_sel = mapped_lon[::desamp]
    mapped_ele_sel = mapped_ele[::desamp]
    azi_sel = azi[::desamp]
    dip_sel = dip[::desamp]

    print(f"Creating Meta data for {selCh.shape[0]} channels")
    print(f"One channel every ~{ChSamp * desamp * desampStrm * 1e-3:.2f} [km]")

    startDate = obspy.UTCDateTime(1926, 1, 1)

    instr_sen = InstrumentSensitivity(
        value=1.0, frequency=1.0,
        input_units="rad", output_units="1/s",
        input_units_description="Measured phase by interrogator unit",
        output_units_description="Strain rate"
    )
    response = Response(instrument_sensitivity=instr_sen)

    equipt = Equipment(
        type="Distributed Acoustic Sensing Unit",
        installation_date=startDate,
        manufacturer=manufacturer,
        model=model,
        description=f"Gauge Length: {GaugeL}[m]; Laser wavelength: {lamdLaser} [nm]"
    )

    inv = Inventory(networks=[], source=source)

    net = Network(
        code=network,
        stations=[],
        description=description,
        start_date=startDate
    )

    for ich in range(selCh.shape[0]):
        sta_code = str(ich).zfill(2)
        sta = Station(
            code=f"{array_name}{sta_code}",
            latitude=mapped_lat_sel[ich],
            longitude=mapped_lon_sel[ich],
            elevation=mapped_ele_sel[ich],
            creation_date=obspy.UTCDateTime(2023, 1, 1),
            start_date=obspy.UTCDateTime(2023, 1, 1),
            site=Site(name=f"{description} {selCh[ich]}/{selChDesamp[ich]}"),
            equipments=[equipt]
        )

        cha = Channel(
            code="HS1",
            location_code="",
            start_date=startDate,
            latitude=mapped_lat_sel[ich],
            longitude=mapped_lon_sel[ich],
            elevation=mapped_ele_sel[ich],
            depth=1.0,
            azimuth=azi_sel[ich],
            dip=dip_sel[ich],
            sample_rate=fs
        )
        cha.response = response
        sta.channels.append(cha)
        net.stations.append(sta)

    inv.networks.append(net)
    inv.write(output_file, format="stationxml", validate=True)
