import rasterio
from rasterio.windows import Window
import numpy as np
import math
from classify import ChunkClassifier

def main(rgb_filepath: str, dsm_filepath: str, model_filepath: str, output_filepath: str, chunk_size: tuple):

    classifier = ChunkClassifier(model_filepath)

    with rasterio.open(rgb_filepath) as rgb_src:
        with rasterio.open(dsm_filepath) as dsm_src:
            meta = dsm_src.meta.copy()

            with rasterio.open(output_filepath, 'w', **meta) as dst:
                # Calculate the number of chunks needed in both dimensions
                n_chunks_x = math.ceil(dsm_src.width / chunk_size)
                n_chunks_y = math.ceil(dsm_src.height / chunk_size)
                print(meta)

                for i in range(n_chunks_x):
                    for j in range(n_chunks_y):

                        print(f'Processing chunk {i}-{j}')
                        
                        window_height = min(chunk_size, dsm_src.height - chunk_size*j)
                        window_width = min(chunk_size, dsm_src.width - chunk_size*i)
                        # Calculate the window position and size for this chunk
                        window = Window(i * chunk_size, j * chunk_size, window_width, window_height)

                        # Read a chunk from the source GeoTIFF
                        rgb_data = rgb_src.read(window=window)
                        dsm_data = dsm_src.read(window=window)

                        # Process the chunk
                        processed_data = classifier.process(rgb_data, dsm_data)
                        
                        # Write the processed chunk to the output GeoTIFF
                        dst.write(processed_data, window=window)

if __name__ == "__main__":
    rgb_filepath = 'data/CS-103A/CS103A_hp_transparent_mosaic_group1.tif'
    dsm_filepath = 'data/CS-103A/CS103A_hp_dsm.tif'
    model_filepath = 'model.pkl'
    output_filepath = 'op.tif'
    chunk_size = 2048  # Set your chunk size here (width, height)

    main(rgb_filepath, dsm_filepath, model_filepath, output_filepath, chunk_size)