import os
import platform
from qgis.PyQt.QtWidgets import QMessageBox, QProgressBar, QInputDialog, QLineEdit
from qgis.core import Qgis as QGis, QgsApplication
from qgis.PyQt.QtCore import QDate
from subprocess import Popen, PIPE
from urllib.request import urlretrieve, build_opener, install_opener
import requests
import site
import pkg_resources


try:
    from pydevd import *
except ImportError:
    None

def install_libs(path_req):
    print(path_req)
    if not os.path.isfile(path_req):
        print(f'no requiriment file in {path_req}')
    
    try:
        p = Popen(["python3","-m","pip", "install","-r",path_req], stdout=PIPE, stderr=PIPE)
        output, errors = p.communicate()
        print(output)
    except Exception as err:
        print(f'error pip{err}')
    return

def create_message(title,message):
        msg = QMessageBox()
        msg.setWindowTitle(f'{title}')
        msg.setText(f'{message}')
        msg.addButton(QMessageBox.Yes)
        msg.addButton(QMessageBox.No)
        ret = msg.exec_()
        return ret

def get_data_from_catalog(self):
        id = self.comboBox_sat_json.currentText()
        sats = self.comboBox_sat_bands_json.currentText()
        req_sat = requests.get(f'https://earthengine-stac.storage.googleapis.com/catalog/{id}/{sats}.json')
        request_collection = req_sat.json()
        data = {'init_date':request_collection["extent"]["temporal"]["interval"][0][0].split('T')[0].split('-'),
                'end_date':request_collection["extent"]["temporal"]["interval"][0][1].split('T')[0].split('-'),
                'id':request_collection['id'],
                'prov':request_collection['providers'],
                'bands':request_collection['summaries']['gee:visualizations'],
                'eoBands':request_collection['summaries']['eo:bands'],
                'gsd':request_collection['summaries']
                }
        print("get_data_from_catalogs")      
        return data

def install_external_libs():
    try:
        
        package_dir = QgsApplication.qgisSettingsDirPath() + "python/plugins/download_raster/"
        requirement_file = os.path.join(package_dir,'requirements.txt')
        string=''
        with open(requirement_file,'r') as f:
            lines=f.readlines()
            for l in lines:
                string = string+l
        
        if create_message("instalacion libs","se instalaran las siguientes librerias requeridas \n"+string) == QMessageBox.Yes:
            install_libs(requirement_file)
    except Exception as err:
        print(f'{err}')

def new_windget_config(self,data):
        print('utils new widg')
        min_d = data['init_date']
        max_d = data['end_date']
        d = QDate(int(min_d[0]),int(min_d[1]),int(min_d[2]))
        df = QDate(int(max_d[0]),int(max_d[1]),int(max_d[2]))
        self.dateEdit_ini.setMinimumDate(d)
        self.dateEdit_ini.setMaximumDate(df)
        self.dateEdit_ini.setDate(d)
        self.dateEdit_end.setMinimumDate(d)
        self.dateEdit_end.setMaximumDate(df)
        self.dateEdit_end.setDate(df)

def valid_data(self):
    
    count_bands = self.selected_bands.count() #count of select bands
    extend_bbx = self.ones.outputExtent() #bounding box extents 
    out_path = self.output_line.text() #output path tif
    correct_bbx = False 
    correct_out_path = False
    bands_is_empy = False
    xm = extend_bbx.xMinimum()
    ym = extend_bbx.yMinimum()
    xma= extend_bbx.xMaximum()
    yma= extend_bbx.yMaximum()
    

    if xm !=0 and ym !=0 and xma !=0 and yma !=0: #validate current extend
        correct_bbx = True
    else :
        print(f'bad bbx')
        create_message('Error BBX','Seleccione un bounding box correcto')

    if len(out_path) != 0: #validate output path
        if os.path.exists(out_path):
            correct_out_path = False
            create_message("Error Directorio",f'El archivo ya existe {out_path}')
        else:
            correct_out_path = True
    else:
        create_message('Error directorio','Formato de directorio incorrecto')

    if count_bands !=0: #validate number bands
        bands_is_empy = True
    else:
        create_message('Error Bandas','Seleccione alguna banda')

    if correct_out_path and correct_bbx and bands_is_empy : 
        return True

def init_config(self):
        name_path = os.path.expanduser("~").replace("\\","//")+'//default.tif'
        self.output_line.setText(name_path) 

def validate_selected_bands(self,bands_):
    for bs in range(self.selected_bands.count()):
        if self.selected_bands.item(bs).text() == bands_:
            return True
        

