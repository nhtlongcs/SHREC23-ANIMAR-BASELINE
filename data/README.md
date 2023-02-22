1. Download the blender tar file, version 2.79 (exclusive). Untar and check the `blender` executive inside.

```bash
$ wget https://download.blender.org/release/Blender2.79/blender-2.79-linux-glibc219-x86_64.tar.bz2
$ tar xjf blender-2.79-linux-glibc219-x86_64.tar.bz2
$ mv blender-2.79-linux-glibc219-x86_64 blender-2.79
```
2. Generate render images 

```bash
blender-2.79/blender -b -P generate_ring.py -- data/ANIMAR_Preliminary_Data/3D_Models
```

It will create a checkpoint file `~/.shrec/save.txt` as it runs, this file stores the index of the last generated object to resume in case of errors occuring. This file needs to be deleted between generation of each phase (else it will keep resuming).
We also provide `bounding` parameter to limit the number of objects to generate. This is useful for debugging purposes (default: -1, i.e. no limit), it could be set to N to generate next N objects (since the last checkpoint in `~/.shrec/save.txt`). This option usefull when you want to generate a subset of the data in multiple devices.

A directory `generated_models` is created next to the `3D_Models` directory. It contains the generated images for each ring view. The directory tree should look like this:

```
data/
├─ ANIMAR_Preliminary_Data/
│  ├─ 3D_Models/
│  │  ├─ *.obj
│  ├─ generated_models/
│  │  ├─ ring#                      
│  │  │  ├─ <type>/                 
│  │  │  │  ├─ Image####.png
```

where
- `rid`: a number (default: 0-6) as ring identifier
- `type`: `depth`, `mask`, or `render`
- `####`: a number (with leading 0s, default: 0001-0012) as view identifier
