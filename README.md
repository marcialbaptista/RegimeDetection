## Welcome to Baselining Repository

Changing operating conditions and environmental conditions pose significant challenges for many analytical tasks, including prognostics. In the presence of dynamic regimes, degradation can have distinct data profiles depending on the operating and environmental conditions. When these conditions are not factored out, it can be difficult to observe the deterioration path of the equipment. Therefore, it is useful to "baseline" the  data to focus on changes of system health. By baselining, we mean to eliminate the extra dimension of the data introduced by the operational and environmental conditions.

This repository provides code to baseline the data of C-MAPSS dataset 2 and 4 using a Self-Organizing Map (SOM) network. The method works without the need to explicitly state the number of operating modes/regimes.   

The basic usage is:
```
bmus_indexes, match_percentages = detect_regimes(dataset_id=2, dimension=20)
```

The `dataset_id` parameter indicates the dataset (2 or 4) of the C-MAPSS repository that you may want to use. The parameter `dimension` allows defining the size of the network grid. 

The function `detect_regimes` return a list of modes for each data point and the list of matching rate for each observed engine unit. 

### Note

The code was developed for C-MAPSS datasets but it can be easily adapted to other applications. 

### Support or Contact

Having trouble with this repository? Contact me and we’ll help you sort it out.
