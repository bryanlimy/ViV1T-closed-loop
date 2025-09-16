## Extract the most and least exciting center and surround grating combination
- inference trained model on all dynamic center and surround gratings with 8 directions each.
  ``` 
  python predict_center_surround_gratings.py --output_dir=<model checkpoint>
  ```
- store the most exciting grating center, the most exciting surround grating given the most exciting center grating, and the least exciting surround grating given the most exciting center grating into a single parquet file
  ```
  python extract_most_exciting_center_surround_grating.py
  ```