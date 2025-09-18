## Extract the most exciting natural center
- inference trained model on all dynamic natural movies centers from the sensorium 2023 training set.
  ``` 
  python predict_natural_center.py --output_dir=<model checkpoint>
  ```
- inference trained model on all static natural movies centers from the sensorium 2023 training set.
  ``` 
  python predict_natural_center.py --output_dir=<model checkpoint> --static
  ```
- store the most exciting static and dynamic natural center in a single parquet file
  ```
  python extract_most_exciting_natural_center.py
  ```