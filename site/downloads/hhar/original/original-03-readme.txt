This is a documentation for the Heterogeneity Dataset for Human Activity Recognition (HHAR) from Smartphones and Smartwatches from the public repository: https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition+Data+Set or the personal Website: http://cs.au.dk/~allans/heterogenity/.

The Heterogeneity Dataset for Human Activity Recognition from Smartphone and Smartwatches is a dataset devised to benchmark human activity recognition algorithms (classification, automatic data segmentation, sensor fusion, feature extraction, etc) containing sensor heterogeneities.
The files in this archive contain all the samples from the still experiment experiment. 
The dataset contains the readings of a motion sensor commonly found in smartphones' recorded while the phones were laying still in one of each of the 6 possible orientation. The files here contains all the samples
from the static still experiment. The data set is structured in the following way:

------------- Static Accelerometer Samples ------------


The data recordings from each specific device are accessible in the following structure : Orientation/DeviceName/DeviceModel.csv
,where 'Orientation' refers to the orientations of the devices in which data was recorded. These orientations are the following: 
	Phoneonback:	The phone is laying on the back of the phone with the screen pointing up (away from the ground).
	Phoneonfront:	The phone is laying on the back of the phone with the screen pointing towards the ground 
	Phoneonbottom:	The phone is standing on the bottom of the screen, meaning the bottom is pointed towards the ground
	Phoneontop:	The phone is standing on the top of the screen, meaning the top is pointed towards the ground
	Phoneonleft:	The phone is laying on the left side of the screen. 
	Phoneonright:	The phone is laying on the right side of the screen.
	

Example: to access the samples from the device named 3Renault-AH of the model Samsung-Galaxy-S3 Mini when laying static on the back we get the following structure:
Phoneonback/3Renault-AH/Samsung-Galaxy-S3 Mini.csv.

Each CSV file consist of 6 columns, holding attributes creation time, sensor time,arrival time,x,y,z, respectively.
The recordings from the three axes from the accelerometer are the x,y,z columns.
For the semantics of the different timestamp versions, see the publication [1].


[1] Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor Siiger Prentow, Mikkel Baun Kj¾rgaard, Anind Dey, Tobias Sonne, and Mads M¿ller Jensen "Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition" In Proc. 13th ACM Conference on Embedded Networked Sensor Systems (SenSys 2015), Seoul, Korea, 2015. http://dx.doi.org/10.1145/2809695.2809718