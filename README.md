# SOAI-Project
We have a video of traffic driving on a road.
The aim is to determine the nature of the vehicles.
Based on YOLO, we already get classes of vehicles (car, truck ...)
However, we would like to have a finer list of classes, following the Swiss10 definition.
For this, we have validation data taken alongside the video with a laser that determines the exact category of vehicle.
There is a timestamp corresponding to the appearance of the vehicle.

The pipeline should go as follows:
- [x] Run Yolo on the video and for each frame the vehicles
- [x] Determine the cluster defining the lanes of the road
- [x] Determine the LOI (Line Of Interest) : determine the timestamp to compare with the truth dataset.
- [ ] Determine the start time : Shift between the start of the video and the recording of the truth.
- [ ] Code and train the transfer learning
- [ ] Run the overwriting of the vehicle class
