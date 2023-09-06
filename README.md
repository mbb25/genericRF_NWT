# Lichen Pixel Prediction

Most of the work here is done with jupyter notebooks. Here's an overview of their functionality:

- 1_data_extraction.ipynb
  - Ingests shapefiles and rasters from the raw data, and extracts pixel data from them. This will include R/G/B data, class (and class certainty), and heightmap category. The resulting dataframe is output as a `.csv` file
- 2_process_training_data.ipynb
  - Ingests the `.csv` output from the first notebook, and creates derived columns such as luminance, chrominance, and other features. This notebook has remained largely unchanged from its original form.
- 2_process_validation_site.ipynb
  - This is only used if you want to run inference on every pixel in a site (for qualitative evaluation, for example). A site name is specified at the top of the notebook, and the script then splits the site into manageable chunks. These chunks are output as pickle files, as this is easier to manage for massive tensors with minimal overhead.
- 3_training_and_inference.ipynb
  - There are custom cross-validation and random search algorithms here. This is to account for the spatial autocorrelation of this data; lichen pixels within one site might look much more similar than between two different sites. To counter this, we do cross-validation by holding out different subsets of the sites.
- 4_composite_clean.ipynb
  - I actually have not touched this one, so I'm not 100% sure if it's still usable
