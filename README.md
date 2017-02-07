# CarND-Behavioral-Cloning

Data Collection
Collecting good driving data from the simulator proved to be one of the bigger challenges of this project. I attempted several times to collect data with the original simulator (keyboard arrows only), and could not produce a strong enough model for the car to stay on the track. Ultimately, the new beta simulator allowed me to successful train the model. A large fraction (40-50%) of the collected data involves error correction - getting the car to move back to the center of the road once it swerves towards the shoulder. 


Preprocessing the images
I applied 4 image preprocessing steps:
  - Scaled down the images from 320x160 to 40x20 - this allowed me to train the model faster without losing much information.
  - grayscaled the images. Color does not play much of a factor on this particular road - though it could in other driving scenarios. Ignoring color simplifies the model and allows it to train slightly quicker
  - Normalize the features 
  - Randomly flip images horizontally (and the associated steering angle). Because most of the turns on the track are left turns, this will help the model avoid a left turn bias.

Model
