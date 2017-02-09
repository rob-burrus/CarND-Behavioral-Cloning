# CarND-Behavioral-Cloning

Data Collection

Collecting good driving data from the simulator proved to be one of the bigger challenges of this project. To start, I collected several laps of driving down the middle of the track. I then spent a lot of time recording "recovery" data, where I would start recording data when the car was already on the side of the track and I would steer it back to the middle. However, training the model with this data resulted in erratic steering controls. The car could hardly make it around the first turn. Ultimately, I recorded 6 laps of driving down the center of the track and utilized the left and right camera angle pictures to augment my dataset. This strategy proved far more successful.  



Model
