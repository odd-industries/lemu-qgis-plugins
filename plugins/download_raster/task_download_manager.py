#import random
#from time import sleep
from qgis.core import QgsApplication, QgsTask, QgsMessageLog
from qgis.PyQt.QtWidgets import *
from .downloader_qgis import GEEDownloader
import ee

from .scripts.utils import create_message


try:
    from pydevd import *
except ImportError:
    None

def new_download(task,date_in,date_finish,out_url,bbx,bar,sat,cloudcover,btn_out,bands,pros,resolution):
    sat_collection = sat.find("LANDSAT")
    if resolution == None:
         resolution = 100

    if sat_collection >= 0:

        cloud_type = 'CLOUD_COVER'

        if pros == 'mean':
            img = ee.ImageCollection(sat)\
                .filterDate(date_in,date_finish)\
                .filter(ee.Filter.lt(cloud_type,cloudcover))\
                .select(bands).mean()
        if pros == 'mosaic':
            img = ee.ImageCollection(sat)\
                .filterDate(date_in,date_finish)\
                .filter(ee.Filter.lt(cloud_type,cloudcover))\
                .select(bands).mosaic()
        if pros == 'first':
            img = ee.ImageCollection(sat)\
                .filterDate(date_in,date_finish)\
                .filter(ee.Filter.lt(cloud_type,cloudcover))\
                .select(bands).first()
    else:
        if pros == 'mean':
            img = ee.ImageCollection(sat)\
                    .filter(ee.Filter.date(date_in, date_finish))\
                    .select(bands).mean()
        if pros == 'mosaic':
            img = ee.ImageCollection(sat)\
                    .filter(ee.Filter.date(date_in, date_finish))\
                    .select(bands).mosaic()
        if pros == 'first':
            img = ee.ImageCollection(sat)\
                    .filter(ee.Filter.date(date_in, date_finish))\
                    .select(bands).first()
                
    GEEDownloader.export_to_local(img,out_url,
                                    region=bbx,
                                    scale=int(resolution),
                                    crs="EPSG:4326",
                                    max_divisions_or_level=10,
                                    parallel=False,
                                    num_workers=10,
                                    progress_fun=bar,
                                    btn_out=btn_out)
    
    if task.isCanceled():
        return None
    
    return {"task": task.description()} 

def completed(task):
    """_summary_
    """
    create_message("Descarga completada","Descarga Completa")
    QgsMessageLog.logMessage(f'completed')
#------simple task test -------#
def test_task(task,numb):
    while numb > 1:
         print(numb)
         numb = numb -1
         
    if task.isCanceled():
        print("asd")
        return None
    return {"task": task.description()} 

def end_task(task):
     print("test")

def create_test_task():
    
    globals()['task1']= QgsTask.fromFunction('my task!',test_task,
                                 numb=100,
                                on_finished=completed
                                 )
    
    QgsApplication.taskManager().addTask(globals()['task1'])
    
#--------end simple task test-------------#


def createTask(date_in,date_finish,out_url,bbx,bar_self,sat,cloudcover,btn_out,bands,pros,resolution):
    """_summary_

    Args:
        date_in (_type_): _description_
        date_finish (_type_): _description_
        out_url (_type_): _description_
        bbx (_type_): _description_
    """
    
    if resolution == '':
        resolution = 100
        create_message('Err','resolucion default 100')
    
    globals()['task1'] = QgsTask.fromFunction('download Raster',new_download,
                                 on_finished=completed,

                                 date_in=date_in,
                                 date_finish=date_finish,
                                 out_url=out_url,
                                 bbx=bbx,
                                 bar=bar_self,
                                 sat=sat,
                                 cloudcover=cloudcover,
                                 btn_out=btn_out,
                                 bands=bands,
                                 pros=pros,
                                 resolution=resolution)
    
    QgsApplication.taskManager().addTask(globals()['task1'])
    

def cancel():
    QgsApplication.taskManager().cancelAll()
    QgsMessageLog.logMessage(f'cancels')
    

