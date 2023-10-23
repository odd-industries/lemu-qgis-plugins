import os
import queue
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union
from numbers import Number

import ee
import numpy as np
import requests
import retry
from requests import HTTPError

import subprocess
from osgeo import gdal
from qgis import processing
from qgis.core import QgsMessageLog, Qgis
import platform
try:
    from pydevd import *
except ImportError:
    None


def merge_tiffs_gdal(tiff_pth_list: List[Union[str, bytes, os.PathLike]],
                     out_file_pth: Union[str, bytes, os.PathLike],
                     out_data_type: str = None,
                     a_nodata: float = None,
                     btn_out = None):
    """
    Merge a list of rasters into one using gdal_merge function.
    This utility will automatically mosaic a set of images. All the images
    must be in the same coordinate system and have a matching number
    of bands, but they may be overlapping, and at different resolutions.
    In areas of overlap, the last image will be copied over earlier ones.
    Nodata/transparency values are considered on a band by band level,
    i.e. a nodata/transparent pixel on one source band will not set a
    nodata/transparent value on all bands for the target pixel in the
    resulting raster nor will it overwrite a valid pixel value.
       See: https://gdal.org/programs/gdal_merge.html

    Args:
        tiff_pth_list (List[Union[str, bytes, os.PathLike]]): List of paths of
            rasters to merge.
        out_file_pth (Union[str, bytes, os.PathLike]): Output path of merged
            raster.
        out_data_type (str, optional): Force the output image bands to have a
            specific data type supported by the driver, which may be one of
            the following: Byte, UInt16, Int16, UInt32, Int32, Float32,
            Float64, CInt16, CInt32, CFloat32 or CFloat64. If None, uses
            a inferred data type from input rasters. Defaults to None.
        a_nodata (float, optional): Value to identify nodata in the raster
            tiles. In the final raster, this value is replaced with nodata.
            If a_nodata=None, no replacement is done.
            Defaults to None.
    """    
    command = 'gdal_merge.py '
    if platform.system() == "Windows":    
        command = 'gdal_merge '

    command += '-of GTiff '
    command += '' if a_nodata is None else f'-a_nodata {a_nodata} '
    command += '-co COMPRESS=ZSTD '
    # PREDICTOR doesn't work with float64
    if out_data_type not in ['Float64', 'float64']:
        command += '-co PREDICTOR=2 '
    command += '-co TILED=YES '
    command += '-co BIGTIFF=YES '
    command += '-co NUM_THREADS=ALL_CPUS '
    # It doesn't work if this is defined before
    command += '' if out_data_type is None else f'-ot {out_data_type} '
    command += f'-o {str(out_file_pth)} '
    command += ' '.join([str(tiff) for tiff in tiff_pth_list])
    output = subprocess.run(command, shell=True) #cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    QgsMessageLog.logMessage(str(output), 'GEEDownloader Plugin', Qgis.Info)
    btn_out.setEnabled(True)

def merge_tiffs_gdal_wrap(tiff_pth_dic: Dict[str, List[os.PathLike]],
                          out_file_pth: Union[str, bytes, os.PathLike],
                          out_data_types: dict = None,
                          band_names: list = None,
                          file_per_band: bool = False,
                          a_nodata: float = None,btn_out = None):
    """
    Wrapper of the merge_tiffs_gdal function. The wrapper controls the
    different ways to merge the tiles if the file_per_band is True or False.

    Args:
        tiff_pth_dic (dict[str, list[os.PathLike]]):  Dictionary with the paths
            to the downloaded images. If file_per_band=True, the keys of the
            dictionary are the names of the bands in the ee.Image. Each key
            is associated with a list of paths to the downloaded GeoTiffs
            of the corresponding band. If file_per_band=False, the
            dictionary has only one key, ‘band’, because the GeoTiffs
            contain all the bands.
        out_file_pth (Union[str, bytes, os.PathLike]): Output path of merged
            raster.
        out_data_types (dict, optional): Force the output image bands to have a
            specific data type supported by the driver, which may be one of
            the following: Byte, UInt16, Int16, UInt32, Int32, Float32,
            Float64, CInt16, CInt32, CFloat32 or CFloat64. If None, uses
            a inferred data type from input rasters. Defaults to None.
        band_names (list): List with the names of the bands in the
            image. Defaults to None.
        file_per_band (bool, optional): Whether to produce a different
            GeoTIFF per band. Defaults to False. Defaults to False.
        a_nodata (float, optional): Value to identify nodata in the raster
            tiles. In the final raster, this value is replaced with nodata.
            If a_nodata=None, no replacement is done.
            Defaults to None.

    Raises:
        Exception: If the length of band_names do not match the number of keys
            in tiff_pth_dic and file_per_band=True.

    Returns:
        None.

    """
    out_file_pth = Path(out_file_pth)
    # File per band
    if file_per_band:
        if len(tiff_pth_dic.keys()) != len(band_names):
            raise Exception(
                f'{len(band_names)} deffined but {len(out_file_pth.keys())} bands were downloaded')
        suffix = out_file_pth.suffix
        if suffix in ['.tif', '.tiff']:
            out_file_pth = out_file_pth.with_suffix('')
        else:
            suffix = '.tif'
        for band_name in band_names:
            out_file_pth_aux = out_file_pth.parent / (out_file_pth.name +
                                                      '.' + band_name + suffix)
            merge_tiffs_gdal(tiff_pth_dic[band_name],
                             out_file_pth_aux,
                             out_data_type=out_data_types[band_name],
                             a_nodata=a_nodata,btn_out = btn_out)
            # Set band names
            raster_ds = gdal.Open(str(out_file_pth_aux), gdal.GA_Update)
            # num_bands = ds.RasterCount
            rb = raster_ds.GetRasterBand(1)
            rb.SetDescription(band_name)
            raster_ds = None
    else:
        if len(tiff_pth_dic.keys()) > 1:
            raise Exception(
                'file_per_band is False, but different files were downloaded')

        merge_tiffs_gdal(tiff_pth_dic[list(tiff_pth_dic.keys())[0]],
                         out_file_pth,
                         out_data_type=out_data_types[list(
                             out_data_types.keys())[0]],
                         a_nodata=a_nodata,btn_out = btn_out)

        # Set band names
        raster_ds = gdal.Open(str(out_file_pth), gdal.GA_Update)
        # num_bands = raster_ds.RasterCount
        for band_idx, band_name in enumerate(band_names):
            rb = raster_ds.GetRasterBand(band_idx + 1)
            rb.SetDescription(band_name)
        raster_ds = None


def rectangle_tesselate(tlbr_rectangle: List[Number],
                        num_divisions: int
                        ) -> Tuple[List[List[Number]], List[List[int]]]:
    """Create a regular tessellation of the input rectangle.

    Args:
        tlbr_rectangle (List[Number]): Input rectangle to tesselate in
            [top, left, bottom, right] format.
        num_divisions (int): Number of regular divisions of the rectangle
            for tesselation

    Returns:
        Tuple[List[List[Number]], List[List[int]]]: Tuple with the list of
        tiles from the tesselation in [top, left, bottom, right] format,
        and the list of matrix-like position of the tiles inside the input
        rectangle, in [i,j] format.
    """
    t, l, b, r = tlbr_rectangle

    dx = (r - l)/num_divisions
    dy = (b - t)/num_divisions
    tile_x0, tile_y0 = l, t

    tiles = []
    positions = []
    for i in range(num_divisions):
        tile_x0 = l
        for j in range(num_divisions):
            tiles.append([tile_y0, tile_x0, tile_y0+dy, tile_x0+dx])
            positions.append((i, j))
            tile_x0 = tile_x0+dx
        tile_y0 = tile_y0+dy
    return tiles, positions


class EE_geometry_tesselator:
    """Iterator class to tesselate an input geometry bounding box."""

    def __init__(self, ee_geometry: ee.Geometry, num_divisions: int):
        """
        Initialize the iterator class to tesselate an input geometry bounding box.

        Args:
            ee_geometry (ee.Geometry): Input ee.Geometry, from which the
                bounding box to be tesseleted is extracted.
            num_divisions (int): Number of regular divisions of the ee.Geometry
                for tesselation (num_divisions X num_divisions).

        Returns:
            None.

        """
        self.ee_geometry = ee_geometry
        self.num_divisions = num_divisions

        lb, rb, rt, lt, _ = ee_geometry.bounds().coordinates().getInfo()[0]
        # top, left, bottom, right
        self.bounds = {'t': lt[1], 'l': lt[0], 'b': rb[1], 'r': rb[0]}

        self.crs = ee_geometry.projection().crs().getInfo()

        # Go from west (rigth) to east (left) and from north (top) to south (bottom).
        self.dx = (self.bounds['r'] - self.bounds['l'])/num_divisions
        self.dy = (self.bounds['b'] - self.bounds['t'])/num_divisions

    def __iter__(self):
        """Initialize an instance of the iterator."""
        self.tile_count = 0
        self.tile_position = [0, 0]
        self.tile_x0 = self.bounds['l']
        self.tile_y0 = self.bounds['t']

        return self

    def __len__(self):
        """Length of the iterator."""
        return self.num_divisions**2

    def __next__(self) -> Tuple[ee.Geometry, List[int]]:
        """
        Get the next tile.

        Raises:
            StopIteration: Python statement to stop the iterator.

        Returns:
            geom (ee.Geometry): The geometry of the tile.
            position (List[int, int]): Position of the tile [y, x], or in 
                geografic orientation [nord-south, west-east].

        """
        if self.tile_count >= self.num_divisions**2:
            raise StopIteration
        else:
            pass

        t = self.bounds['t'] + (self.dy * self.tile_position[0])
        l = self.bounds['l'] + (self.dx * self.tile_position[1])
        b = t + self.dy
        r = l + self.dx

        geom = ee.Geometry.Polygon(
            [
                [
                    [l, b],
                    [r, b],
                    [r, t],
                    [l, t],
                    [l, b],
                ]
            ],
            proj=self.crs, geodesic=False)

        position = self.tile_position.copy()

        self.tile_count += 1
        if self.tile_position[1] == self.num_divisions - 1:
            self.tile_position[1] = 0
            self.tile_position[0] += 1
        else:
            self.tile_position[1] += 1

        return geom, position


class GEEDownloader:
    @classmethod
    def ee_geometry_tesselate(cls,
                              ee_geometry: ee.Geometry,
                              num_divisions: int
                              ) -> Tuple[List[ee.Geometry], List[List[int]]]:
        """Create a regular tessellation of the input ee.Geometry bounding box.

        Args:
            ee_geometry (ee.Geometry): Input ee.Geometry, from which the
                bounding box to be tesseleted is extracted.
            num_divisions (int): Number of regular divisions of the ee.Geometry
                for tesselation.

        Returns:
            Tuple[List[ee.Geometry], List[List[int]]]: Tuple with the list of
            tiles from the tesselation and the list of matrix-like position
            of the tiles inside the bounding box, in [i,j] format.
        """
        lb, rb, rt, lt, _ = ee_geometry.bounds().coordinates().getInfo()[0]

        tiles, positions = rectangle_tesselate([lt[1], lt[0], rb[1], rb[0]],
                                               num_divisions)
        crs = ee_geometry.projection().crs().getInfo()
        tiles_ee = []
        for tile in tiles:
            t, l, b, r = tile
            geom = ee.Geometry.Polygon(
                [
                    [
                        [l, b],
                        [r, b],
                        [r, t],
                        [l, t],
                        [l, b],
                    ]
                ],
                proj=crs, geodesic=False)
            feature = ee.Feature(geom, {})
            tiles_ee.append(feature.geometry())

        return tiles_ee, positions

    @classmethod
    @retry.retry((ConnectionError, RuntimeError), tries=15, delay=1, backoff=2)
    def ee_download_image(cls,
                          ee_object: ee.Image,
                          out_file_pth: Union[str, bytes, os.PathLike],
                          scale: int = None,
                          crs: str = None,
                          region: ee.Geometry = None,
                          file_per_band: bool = False,
                          timeout: int = 300,
                          rewrite: bool = True):
        """Export an ee.Image as a GeoTIFF.

        Args:
            ee_object (ee.Image): The ee.Image to download.
            out_file_pth (Union[str, bytes, os.PathLike]): Output filename for
                the exported image or Path object.
            scale (int, optional): A default scale to use for any bands that do
                not specify one. Defaults to None.
            crs (str, optional): A default CRS string to use for any bands that
                do not explicitly specify one. Defaults to None.
            region (ee.Geometry, optional): A polygon specifying a region to
                download. Defaults to None.
            file_per_band (bool, optional): Whether to produce a different
                GeoTIFF per band. Defaults to False.
            timeout (int, optional): The timeout in seconds for the request.
                Defaults to 300.
            rewrite (bool, optional): Allow rewriting of the output file
                'out_file_pth' if it already exists.
                Defaults to True.

        Raises:
            expt: GEE exception, ee.EEException, while obtaining the url and
                downloading the data..
            ConnectionError: Unexpected exception while obtaining the url and
                downloading the data.
            HTTPError: If the request response is not 200.

        Returns:
            Dict[str, os.PathLike]: Dictionary with the paths to the downloaded
            images. If file_per_band=True, the keys of the dictionary are
            the names of the bands in the ee.Image. Each key is associated
            with the path to the downloaded GeoTiff of the corresponding
            band. If file_per_band=False, the dictionary has only one key,
            'band',because all the bands were downloaded in the same file.
        """

        if not isinstance(ee_object, ee.Image):
            QgsMessageLog.logMessage("The ee_object must be an ee.Image.", 'GEEDownloader Plugin', Qgis.Critical)
            return

        if not rewrite:
            if file_per_band:
                bands_names = ee_object.bandNames().getInfo()
                is_files = []
                for band_name in bands_names:
                    file_pth_tif = out_file_pth.with_suffix(
                        f'.{band_name}.tif')
                    is_files.append(file_pth_tif.is_file())
                if np.all(is_files):
                    tiles_pths = {}
                    for band_name in bands_names:
                        tiles_pths[band_name] = out_file_pth.with_suffix(
                            f'.{band_name}.tif')
                    return tiles_pths
            else:
                file_pth_tif = out_file_pth.with_suffix('.tif')
                if file_pth_tif.is_file():
                    tiles_pths = {}
                    band_name = 'band'
                    tiles_pths[band_name] = file_pth_tif
                    return tiles_pths

        out_file_pth = Path(out_file_pth)
        name = out_file_pth.stem

        # https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl
        params = {'name': name,
                  'filePerBand': file_per_band}

        # NOTE: 'filePerBand' does't work if 'format' is defined. Only can be
        #        defined as the internal value 'ZIPPED_GEO_TIFF_PER_BAND'
        if not file_per_band:
            params['format'] = 'ZIPPED_GEO_TIFF'

        if scale is not None:
            params["scale"] = scale
        else:
            params["scale"] = ee_object.projection().nominalScale()
        if region is None:
            region = ee_object.geometry()
        params["region"] = region
        if crs is not None:
            params["crs"] = crs

        # Errors that are considered for retrying the download.
        list_errors = [
            'Please try again.',                        # 'code': 503
            'Earth Engine memory capacity exceeded.',   # 'code': 503
            'An internal error has occurred',           # 'code': 500
            'Too Many Requests: Request was rejected',  # 'code': 429
            'Computation timed out.']                   # 'code': 400
            
        try:
            url = ee_object.getDownloadURL(params)
            r = requests.get(url, stream=True, timeout=timeout)
        except ee.EEException as expt:
            bool_errors = [x in str(expt) for x in list_errors]
            if np.any(bool_errors):
                QgsMessageLog.logMessage(
                    f'ee.EEException considered for retry occurred while getting url for image {out_file_pth.stem}: {expt}', 'GEEDownloader Plugin', Qgis.Critical)
                raise ConnectionError
            else:
                QgsMessageLog.logMessage(f'ee.EEException while getting url for image {out_file_pth.stem}: {expt}', 'GEEDownloader Plugin', Qgis.Critical)
                raise expt
        except Exception as expt:
            QgsMessageLog.logMessage(
                f'Unexpected exption in getting request of image {out_file_pth.stem}: {expt}', 'GEEDownloader Plugin', Qgis.Critical)
            raise ConnectionError

        if r.status_code != 200:
            msg = r.json()['error']['message']

            bool_errors = [x in msg for x in list_errors]
            if np.any(bool_errors):
                log.warning(
                    f'An error considered for retry occurred while downloading image {out_file_pth.stem}: {r.json()}')
                raise RuntimeError
            else:
                raise HTTPError(
                    f"An error occurred while downloading image {out_file_pth.stem}: {r.json()}")

        with tempfile.TemporaryDirectory() as tmp_dir_pth:

            zip_file = Path(tmp_dir_pth) / f'{name}.zip'
            try:
                with open(zip_file, "wb") as fd:
                    for chunk in r.iter_content(chunk_size=1024):
                        fd.write(chunk)
            except Exception as expt:
                QgsMessageLog.logMessage(
                    f'Unexpected exption in request write of image {out_file_pth.stem}: {expt}')
                raise RuntimeError
            with zipfile.ZipFile(zip_file) as z:
                z.extractall(out_file_pth.parent)

            with zipfile.ZipFile(zip_file) as zip_obj:
                zipinfos = zip_obj.infolist()
            tiles_pths = {}
            for i in range(len(zipinfos)):
                if file_per_band:
                    band_name = zipinfos[i].filename.split('.')[-2]
                else:
                    band_name = 'band'
                tiles_pths[band_name] = out_file_pth.parent / \
                    zipinfos[i].filename
        return tiles_pths

    @classmethod
    def ee_download_image_tiles_threaded(cls,
                                         ee_object: ee.Image,
                                         out_dir_pth: Union[str, bytes,
                                                            os.PathLike],
                                         n_divisions: int,
                                         scale: int = None,
                                         crs: str = None,
                                         region: ee.Geometry = None,
                                         file_per_band: bool = False,
                                         timeout: int = 300,
                                         num_threads: int = 8,
                                         rewrite: bool = True,

                                         progress_fun = None):
        """Download a regular n_divisions tesselation of the input
        ee.Image to a folder using threads. The names of the tiles correspond
        to matrix-like position of the tiles inside the region, in the form
        tile_{i}_{j}.tif

        Args:
            ee_object (ee.Image): The ee.Image to download.
            out_dir_pth (Union[str, bytes, os.PathLike]): Output directory for
                the exported tiles or Path object.
            n_divisions (int): Number of regular divisions of the image for
                tesselation
            scale (int, optional): A default scale to use for any bands that do
                not specify one. Defaults to None.
            crs (str, optional): A default CRS string to use for any bands that
                do not explicitly specify one. Defaults to None.
            region (ee.Geometry, optional): A polygon specifying a region to
                download. Defaults to None.
            file_per_band (bool, optional): Whether to produce a different
                GeoTIFF per band. Defaults to False.
            timeout (int, optional): The timeout in seconds for the request.
                Defaults to 300.
            num_threads (int, optional): Number of threads to use for parallel
                download. Defaults to 8.
            rewrite (bool, optional): Allow rewriting of the output files in
                'out_dir_pth' if they already exist.
                Defaults to True

        Raises:
            HTTPError: If any error ocurrs while downloading the tiles.

        Returns:
            dict[str, list[os.PathLike]]: Dictionary with the paths to the
            downloaded images. If file_per_band=True, the keys of the
            dictionary are the names of the bands in the ee.Image. Each key
            is associated with a list of paths to the downloaded GeoTiffs
            of the corresponding band. If file_per_band=False, the
            dictionary has only one key, ‘band’, because the GeoTiffs
            contain all the bands
        """
        stopped = threading.Event()
        # tiles_pths = []
        tiles_pths = {}
        count = []

        def ee_download_thread(q):
            while True:
                if not stopped.is_set():
                    img, fname, pars = q.get()
                    try:
                        tiles_pths_aux = cls.ee_download_image(img,
                                                               fname, **pars)
                        for item in tiles_pths_aux.items():
                            if item[0] in tiles_pths:
                                tiles_pths[item[0]].append(item[1])
                            else:
                                tiles_pths[item[0]] = [item[1]]

                        # NOTE: D[x] = D[x] + 1  is not thread-safe
                        count.append(1)

                        if progress_fun is not None:
                            progress_fun.setValue(len(count)/(n_divisions*n_divisions)*100)
                        #QgsMessageLog.logMessage(
                        #    (f'Downloaded tile {fname.stem} number {len(count)} of {n_divisions*n_divisions}'), 'GEEDownloader Plugin', Qgis.Info)
                        q.task_done()
                    except Exception as e:
                        QgsMessageLog.logMessage(f'Task for {fname.stem} error: {e}', 'GEEDownloader Plugin', Qgis.Critical)
                        q.task_done()
                        stopped.set()
                else:
                    _ = q.get()
                    q.task_done()

        q = queue.Queue(maxsize=50)
        out_dir_pth = Path(out_dir_pth)

        # Se crean los threads
        for _ in range(num_threads):
            worker = threading.Thread(target=ee_download_thread,
                                      args=(q,),
                                      daemon=True)
            worker.start()

        # Se asignan las tareas hasta que se terminen
        lead_0 = len(str(n_divisions))
        regions_iter = EE_geometry_tesselator(region, n_divisions)
        for tile, pos in regions_iter:
            params = {'scale': scale,
                      'crs': crs,
                      'region': tile,
                      'file_per_band': file_per_band,
                      'timeout': timeout,
                      'rewrite': rewrite}
            file_pth = out_dir_pth / \
                f'tile_{pos[0]:0{lead_0}d}_{pos[1]:0{lead_0}d}_{n_divisions}'
            q.put((ee_object, file_pth, params))
        q.join()
        if len(count) != n_divisions * n_divisions:
            raise HTTPError("An error occurred while downloading the tiles.")

        return tiles_pths

    @classmethod
    def export_to_local(cls,
                        ee_object: ee.Image,
                        out_file_pth: Union[str, bytes, os.PathLike],
                        scale: int = None,
                        crs: str = None,
                        region: ee.Geometry = None,
                        file_per_band: bool = False,
                        timeout: int = 300,
                        tesselated_2: bool = False,
                        initial_divisions_or_level: int = 1,
                        max_divisions_or_level: int = 100,
                        parallel: bool = True,
                        num_workers: int = 4,
                        out_data_type: str = None,
                        a_nodata: Union[int, float] = None,
                        tile_download_dir: Union[str, os.PathLike] = None,
                        rewrite: bool = True,
                        progress_fun = None,btn_out=None):
        
        """Downloads a raster from GEE. To perform the download the raster is
        tessellated into tiles, and each tile is downloaded separately.
        Then, the tiles are merged in the final raster. Different levels of
        tessellation are tested starting from 'initial_division', until a
        successful download is obtained or 'max_divisions' is reached.

        Args:
            ee_object (ee.Image): The ee.Image to download.
            out_file_pth (Union[str, bytes, os.PathLike]): Output filename for
                the exported image or Path object.
            scale (int, optional): Scale to download the image. If scale=None,
                the scale defined in the input ee.Image is used. Note that the
                default scale of the images on Google Earth Engine is 1°,
                111319.490 meters. If you want a different sale in the download
                image, the ‘scale’ parameter must be defined or the scale in
                the input ee.Image must be changed (ee_image.reproject()).
                Defaults to None.
            crs (str, optional): A default CRS string to use for any bands that
                do not explicitly specify one. Defaults to None.
            region (ee.Geometry, optional): A polygon specifying a region to
                download. Defaults to None.
            file_per_band (bool, optional): Whether to produce a different
                GeoTIFF per band. Defaults to False.
            timeout (int, optional): The timeout in seconds for the request.
                Defaults to 300.
            tesselated_2 (bool, optional): Make the tessellation increase in
                2^n the number of divisions in each iteration.
                Defaults to False.
            initial_divisions_or_level (int, optional): If tesselated_2=False,
                it is the initial number of divisions, in height and width, to
                tessellate the raster in tiles. If tesselated_2=True, it is the
                initial level of divisions in which 
                height=width=2^initial_divisions_or_level. If this
                value is too small, greater values will be tested until a
                successful download is obtained or max_divisions is reached.
                Defaults to 1.
            max_divisions_or_level (int, optional): Maximum number of divisions
                or levels to tessellate the raster in tiles. Defaults to 100.
            parallel (bool, optional): To use parallel download. Defaults to
                True.
            num_workers (int, optional): Number of workers to use for parallel
                download. Defaults to 4.
            out_data_type (str, optional): Force the output image bands to have
                a specific data type supported by the driver, which may be one
                of the following: Byte, UInt16, Int16, UInt32, Int32, Float32,
                Float64, CInt16, CInt32, CFloat32 or CFloat64. If None, uses
                a inferred data type from input rasters. Defaults to None.
            a_nodata (int, optional): Value to identify nodata in the raster
                during the download. In the final raster, this value is replaced
                with nodata. If a_nodata=None, the nodata pixels in the original
                ee.Image will be 0 in the downloaded raster.
                WARNING: Be careful of: (1) To choose a valid value in the data
                type of the ee.Image; (2) The a_nodata value must not be ammong
                the valid information of the ee.Image, otherwise these pixels
                will be nodata in the downloaded raster.
                Defaults to None.
                Examples of appropriate values:
                    - If the data type of the ee.Image is int32 and the valid
                        information is in [1,2,3,…], a_nodata can be 0 or -1.
                    - If the data type of the ee.Image is float64, but the
                        valid values are in the uint16 range, a_nodata can
                        takes values outside the range of uint16;
                        a_nodata<0,a_nodata>65535.
            tile_download_dir (Path, optional): Path to tile download directory.
                If not defined, a temporary directory will be used.
                Defaults to None.
            rewrite (bool, optional): Allow rewriting of the output files in
                'tile_download_dir' if they already exist. If False, only the
                missing tiles will be downloaded.
                Defaults to True
        """
        if a_nodata is not None:
            ee_object = ee_object.unmask(a_nodata)

        band_names = ee_object.bandNames().getInfo()

        if tesselated_2:
            tess_list = 2**np.array((range(initial_divisions_or_level,
                                           max_divisions_or_level)))
            tess_list.astype('int')
        else:
            tess_list = range(initial_divisions_or_level,
                              max_divisions_or_level)

        for n in tess_list:
            if tesselated_2:
                QgsMessageLog.logMessage(
                    f'Trying download at n={n}, level={int(np.log2(n))}', 'GEEDownloader Plugin', Qgis.Info)
            else:
                QgsMessageLog.logMessage(f'Trying download at n={n}', 'GEEDownloader Plugin', Qgis.Info)
            lead_0 = len(str(n))
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_pth = Path(tmp_dir) if tile_download_dir is None else Path(
                    tile_download_dir)
                try:
                    if not parallel:
                        tile_count = 0
                        tiles_pths = {}

                        regions_iter = EE_geometry_tesselator(region, n)
                        for tile, pos in regions_iter:
                            tile_pth = tmp_dir_pth / \
                                f'tile_{pos[0]:0{lead_0}d}_{pos[1]:0{lead_0}d}_{n}'
                            out_filepaths = cls.ee_download_image(
                                ee_object,
                                tile_pth,
                                scale=scale,
                                region=tile,
                                crs=crs,
                                file_per_band=file_per_band,
                                timeout=timeout,
                                rewrite=rewrite)
                            for item in out_filepaths.items():
                                if item[0] in tiles_pths:
                                    tiles_pths[item[0]].append(item[1])
                                else:
                                    tiles_pths[item[0]] = [item[1]]
                            tile_count += 1
                            if progress_fun is not None:
                                progress_fun.setValue(tile_count/(n*n)*100)
                            #QgsMessageLog.logMessage(
                            #    (f'Downloaded tile {tile_pth.stem} number {tile_count} of {n*n}'), 'GEEDownloader Plugin', Qgis.Info)

                    else:
                        tiles_pths = cls.ee_download_image_tiles_threaded(
                            ee_object,
                            tmp_dir_pth,
                            n,
                            scale=scale,
                            region=region,
                            crs=crs,
                            file_per_band=file_per_band,
                            timeout=timeout,
                            num_threads=num_workers,
                            rewrite=rewrite,
                            progress_fun=progress_fun)
                    
                    # Compile final raster
                    data_types_dic = {
                        gdal.GDT_Byte: 'Byte',
                        gdal.GDT_UInt16: 'UInt16',
                        gdal.GDT_Int16: 'Int16',
                        gdal.GDT_UInt32: 'UInt32',
                        gdal.GDT_Int32: 'Int32',
                        gdal.GDT_Float32: 'Float32',
                        gdal.GDT_Float64: 'Float64',
                        gdal.GDT_CInt16: 'CInt16',
                        gdal.GDT_CInt32: 'CInt32',
                        gdal.GDT_CFloat32: 'CFloat32',
                        gdal.GDT_CFloat64: 'CFloat64',
                    }
                    out_data_type_aux = {}
                    
                    for layer_key in tiles_pths.keys():
                        if out_data_type is None:
                            one_tile = gdal.Open(str(tiles_pths[layer_key][0]))
                            out_data_type_aux[layer_key] = data_types_dic[one_tile.GetRasterBand(1).DataType]
                        else:
                            out_data_type_aux[layer_key] = out_data_type
                    
                    QgsMessageLog.logMessage("Merging tiles...", 'GEEDownloader Plugin', Qgis.Info)
                    merge_tiffs_gdal_wrap(tiff_pth_dic=tiles_pths,
                                          out_file_pth=out_file_pth,
                                          out_data_types=out_data_type_aux,
                                          band_names=band_names,
                                          file_per_band=file_per_band,
                                          a_nodata=a_nodata,btn_out = btn_out)
                    
                    break

                except Exception as expt:
                    QgsMessageLog.logMessage(f'In try: {expt}', 'GEEDownloader Plugin', Qgis.Critical)
        return
