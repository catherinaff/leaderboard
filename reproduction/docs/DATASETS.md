# Dataset protocol

| Dataset | Classes / subjects | Native window / stride | Channels | Test / validation | Validation selector |
|---|---|---|---|---|---|
| [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | 6 / 30 | 128 / 64, 50Hz | 9 | Official2947 test windows; stratified10% of official train as validation | Accuracy |
| [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | 12 / 8 (101–108) | 200 / 100, 100Hz | 27 | All8 LOSO subjects; next numeric subject for validation | Weighted F1 |
| [USC-HAD](https://sipi.usc.edu/had/) | 12 / 14 | 200 / 100, 100Hz | 6 | All14 LOSO subjects; next numeric subject for validation | Accuracy |
| [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) | 8 / 15 | 100 / 50, 50Hz | 42, seven positions | All15 LOSO subjects; next numeric subject for validation | Macro F1 |

Native labels, sample counts, array fingerprints and exact classes are in `../configs/dataset_fingerprints.json`. Model adapters may use fewer sensor channels while retaining every canonical sample/label. UCI imputation/scaling is fitted after its train/validation split; LOSO holds out separate validation and test subjects. PAMAP monitors Weighted F1 in the shared loops while the displayed overall ranking uses Accuracy. SensorLLM uses its separate Accuracy selector except RealWorld Macro F1.
