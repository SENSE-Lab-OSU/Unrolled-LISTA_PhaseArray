#%% import packages
import h5py
import numpy as np
import json 
from collections import namedtuple
from json import JSONDecoder
import sys
sys.path.insert(1, '/research/nfs_ertin_1/nithin_data/mod/blip/generated/python')
sys.path.insert(2, '/research/nfs_ertin_1/nithin_data/mod/blip/interface/py')
from blip_ScanVolume import ScanVolume
from cg_Buff_CF import Buff_CF


class Process_pipeline:
    def __init__(self, input_dir, output_dir, method,radar_config,nlp_config):
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.method=method
        self.radar_config=radar_config
        self.nlp_config=nlp_config

class nlp_config:
    def __init__(self, range_min, range_max, range_oversample,
                 velocity_min,velocity_max,num_vel,az_width_rad,num_az,el_width_rad,num_el):
        self.range_min=range_min
        self.range_max=range_max
        self.range_oversample=range_oversample
        self.velocity_min=velocity_min
        self.velocity_max=velocity_max
        self.num_vel=num_vel
        self.az_width_rad=az_width_rad
        self.num_az=num_az
        self.el_width_rad=el_width_rad
        self.num_el=num_el

class radar_config:
    def __init__(self, RadarConfigData, DataAssumptions, PreProcessing):
        self.RadarConfigData=RadarConfigData
        self.DataAssumptions=DataAssumptions
        self.PreProcessing=PreProcessing
           
def Process_pipelineDecoder(obj):
    return Process_pipeline(obj['input_dir'], obj['output_dir'], obj['method'],obj['radar_config'],obj['nlp_config'])



# %% load file 
def loadFile(filename_header,filename_data):
    scanHeader = ScanVolume()
    scan_iq = Buff_CF()

    #filename_header = 'E:/darpa BLip data/2308-NOAA-data/spoiledBeam/Data_Collect_Stagger_Same_Number_1_Spoiled_TX_REM_Mat_50_Chan_300us_Cal_Tower_09192023/header_scan_001.bin'
    f = open(filename_header, mode="rb")
    data = f.read()
    scanHeader.binary_deserialize(data)
    f.close()


    #filename_data = 'E:/darpa BLip data/2308-NOAA-data/spoiledBeam/Data_Collect_Stagger_Same_Number_1_Spoiled_TX_REM_Mat_50_Chan_300us_Cal_Tower_09192023/iq_scan_001.bin'
    f = open(filename_data, mode="rb")

    data_iq = f.read()
    f.close()
    numBytesmd5sum = 16
    mdfSumvalue = data_iq[0:numBytesmd5sum]
    data_iq = data_iq[numBytesmd5sum:]

    numBytesDimension = 8
    dims_length = np.frombuffer(data_iq[0:numBytesDimension],dtype='<Q',count=1)

    data_iq = data_iq[numBytesDimension:]
    dims = np.frombuffer(data_iq[0:np.int_(numBytesDimension*dims_length[0])],dtype='<Q',count=np.int_(dims_length[0]))


    data_iq = data_iq[np.int_(numBytesDimension*dims_length[0]):]

    numBytesDimension = 8
    dims_length = np.frombuffer(data_iq[0:numBytesDimension],dtype='<Q',count=1)
    data_iq = data_iq[numBytesDimension:]

    numBytesData = 4
    data_samples = np.frombuffer(data_iq[0:np.int64(numBytesData*dims_length[0]*2)],dtype='<f',count=np.int64(dims_length[0]*2))
    data_samples_complex= data_samples[::2]+1j*data_samples[1::2]

    data_iq = data_iq[np.int64(numBytesData*dims_length[0]*2):]
    return data_samples_complex,scanHeader

# %%
def get_cpi_samples(data_samples_complex,CPIHeader):
    num_samples = np.prod(CPIHeader.Dims)
    cpi_iq = np.reshape(data_samples_complex[CPIHeader.IQStart + np.arange(num_samples)],CPIHeader.Dims)
    return cpi_iq
# %% main module test
if __name__ == "__main__":
    fileJson = open('/research/nfs_ertin_1/nithin_data/mod/blip/config/process/process_Static_Tower_09192023.json')
    Process_pipelineObj = json.load(fileJson,object_hook=Process_pipelineDecoder)

    fileJsonradar = open(Process_pipelineObj.radar_config)
    fileJsonnlp = open(Process_pipelineObj.nlp_config)


    nlp_configObjDict = json.load(fileJsonnlp)
    nlp_configObj = nlp_config(nlp_configObjDict['range_min'], nlp_configObjDict['range_max'],
                            nlp_configObjDict['range_oversample'],nlp_configObjDict['velocity_min'],
                            nlp_configObjDict['velocity_max'],nlp_configObjDict['num_vel'],
                            nlp_configObjDict['az_width_rad'],nlp_configObjDict['num_az'],
                            nlp_configObjDict['el_width_rad'], nlp_configObjDict['num_el'])
    radar_configDict = json.load(fileJsonradar)
    radar_configObj = radar_config(radar_configDict['RadarConfigData'],
                                radar_configDict['DataAssumptions'],
                                radar_configDict['PreProcessing'])


    filename_header = 'E:/darpa BLip data/2308-NOAA-data/spoiledBeam/Data_Collect_Stagger_Same_Number_1_Spoiled_TX_REM_Mat_50_Chan_300us_Cal_Tower_09192023/header_scan_001.bin'
    filename_data = 'E:/darpa BLip data/2308-NOAA-data/spoiledBeam/Data_Collect_Stagger_Same_Number_1_Spoiled_TX_REM_Mat_50_Chan_300us_Cal_Tower_09192023/iq_scan_001.bin'

    data_samples_complex,scanHeader = loadFile(filename_header,filename_data)
    #%% 
    cpi_iq = get_cpi_samples(data_samples_complex,scanHeader.CPIHeaders[0])
