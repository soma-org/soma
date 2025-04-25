# Workers

Workers are actors that control access and scheduling to individual resources. The resource can be anything from CPU bound work like encryption, to OS limited filesystem access, to concurrent downloading of data. Workers manage their own concurrency and schedule work to be done on a given resource. 