"""
@author: Andrew Smith 
contributors: Valerie Desnoux, Matt Considine
Version 24 July 2023

"""
import numpy as np
import cv2 #MattC
import os
import mmap

class video_reader:

    def __init__(self, file, buffer_size = 25):
        # ouverture et lecture de l'entete du fichier ser
        self.file = file
        self.buffer_size = buffer_size
        self.buffer_remaining = 0
        
        if self.file.upper().endswith('.SER'): #MattC 20210726
            self.SER_flag=True
            self.AVI_flag=False
        elif self.file.upper().endswith('.AVI'):
            self.SER_flag=False
            self.AVI_flag=True
            self.infiledatatype='uint8'
        else:
            raise Exception('error input file ' + file + 'neither is SER nor AVI')
        
        #ouverture et lecture de l'entete du fichier ser

        if self.SER_flag: #MattC
            self.FileID=np.fromfile(file, dtype='int8',count=14)
            offset=14

            self.LuID=np.fromfile(file, dtype=np.uint32, count=1, offset=offset)
            offset=offset+4
        
            self.ColorID=np.fromfile(file, dtype='uint32', count=1, offset=offset)
            offset=offset+4
        
            self.littleEndian=np.fromfile(file, dtype='uint32', count=1,offset=offset)
            offset=offset+4
        
            self.Width=np.fromfile(file, dtype='uint32', count=1,offset=offset)[0]
            offset=offset+4

            self.Height=np.fromfile(file, dtype='uint32', count=1,offset=offset)[0]
            offset=offset+4

            PixelDepthPerPlane=np.fromfile(file, dtype='uint32', count=1,offset=offset)
            self.PixelDepthPerPlane=PixelDepthPerPlane[0]
            offset=offset+4

            FrameCount=np.fromfile(file, dtype='uint32', count=1,offset=offset)
            self.FrameCount=FrameCount[0]
            offset=offset+4

            if self.PixelDepthPerPlane==8:
                self.infiledatatype='uint8'
                self.count=int(self.Width)*int(self.Height)       # Nombre d'octet d'une trame
                self.infilebytes=1
            else:
                self.infiledatatype='uint16'
                self.count=int(self.Width)*int(self.Height)      # Nombre d'octet d'une trame
                self.infilebytes=2
            self.FrameIndex=-1             # Index de trame, on evite les deux premieres
            self.offset=178               # Offset de l'entete fichier ser
            self.fileoffset=178 #MattC to avoid stomping on offset accumulator
            
        elif self.AVI_flag: #MattC 
    	    #deal with avi file
            self.file = cv2.VideoCapture(file)

            self.Width = int(self.file.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.Height = int(self.file.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.PixelDepthPerPlane=1*8
            self.FrameCount = int(self.file.get(cv2.CAP_PROP_FRAME_COUNT))            
            self.count=int(self.Width)*int(self.Height)
            self.infilebytes=1
            self.FrameIndex=-1
            self.offset = 0
            self.fileoffset = 0 #MattC to avoid stomping on offset accumulator
        else: #MattC
    	    ok_flag = False

        if self.Width > self.Height:
            self.flag_rotate = True
            self.ih = self.Width
            self.iw = self.Height
        else:
            self.flag_rotate = False
            self.iw = self.Width
            self.ih = self.Height
        #print(f'in video reader with nframes, height, width = {self.FrameCount}, {self.ih}, {self.iw}')

    def next_frame(self):
        self.FrameIndex += 1
        self.offset = int(self.fileoffset) + self.FrameIndex * int(self.count) * int(self.infilebytes) #MattC track offset
      
        if self.SER_flag: #MattC
            if self.buffer_remaining == 0:
                self.buf = np.fromfile(
                    self.file,
                    dtype = self.infiledatatype,
                    count = self.count * max(0, min(self.buffer_size, self.FrameCount - self.FrameIndex)),
                    offset=self.offset)
                self.buffer_remaining = self.buffer_size
            i = self.buffer_size - self.buffer_remaining
            img = self.buf[self.count*i : self.count*(i+1)]
            self.buffer_remaining -= 1

            
        elif self.AVI_flag:
            ret, img = self.file.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise Exception('error input file is neither is SER nor AVI')

        img = np.reshape(img, (self.Height, self.Width))
        
        if self.flag_rotate:
            img = np.rot90(img)
        if self.infiledatatype == 'uint8':
            img = np.asarray(img, dtype='uint16')*256 #upscale 8-bit to 16-bit
        return img

    def has_frames(self):
        return self.FrameIndex + 1 < self.FrameCount

# ---------------------------------------------------------------------------
# MmapSERReader: 把 SER 文件直接 mmap 进来,以 numpy 视图方式按帧回放
# ---------------------------------------------------------------------------
# 设计要点:
#   * 不写额外文件,直接把 5+ GB 的 SER mmap 进进程虚拟地址空间
#   * OS 自动把按页访问的热点数据缓存在系统文件缓存中,第二遍零成本
#   * 与 video_reader 同样的 has_frames/next_frame 接口
#   * 兼容 8-bit SER (infilebytes=1)
class MmapSERReader:
    """mmap-backed reader for SER files.  Identical read-side interface to
    video_reader (has_frames, next_frame, ih/iw/Width/Height/FrameCount) but
    pulls frames from an mmap rather than re-reading the file with np.fromfile.
    """

    def __init__(self, path, dtype='uint16'):
        # Parse the SER header using the same code path as video_reader.
        meta = video_reader(path)
        self.ih = meta.ih
        self.iw = meta.iw
        self.Width = meta.Width
        self.Height = meta.Height
        self.FrameCount = int(meta.FrameCount)
        self.infiledatatype = meta.infiledatatype
        self.flag_rotate = meta.flag_rotate
        self.FrameIndex = -1
        itemsize = 1 if self.infiledatatype == 'uint8' else 2
        # mmap the file
        self._fd = os.open(path, os.O_BINARY | os.O_RDONLY)
        file_size = os.fstat(self._fd).st_size
        # 178 bytes SER header
        header_size = 178
        self._mm = mmap.mmap(self._fd, file_size, access=mmap.ACCESS_READ)
        # Build the numpy view; clip the trailing bytes if the file does
        # not end on a clean frame boundary.  Reshape using the raw
        # (Height, Width) order from the SER header -- the rotation is
        # applied on every frame in next_frame(), mirroring video_reader.
        data = np.frombuffer(self._mm, dtype='uint8' if itemsize == 1 else 'uint16',
                             offset=header_size)
        per_frame_pixels = int(meta.Height) * int(meta.Width)
        usable = data.size // per_frame_pixels
        usable = min(usable, self.FrameCount)
        usable_pixels = usable * per_frame_pixels
        self._arr = data[:usable_pixels].reshape(usable,
                                                 int(meta.Height),
                                                 int(meta.Width))
        # Drop meta-reader so its file handle is released.
        del meta
        # Truncate FrameCount if the file was shorter than declared.
        self.FrameCount = int(self._arr.shape[0])

    def has_frames(self):
        return self.FrameIndex + 1 < self.FrameCount

    def reset(self):
        """Rewind to the first frame so the same mmap can be replayed."""
        self.FrameIndex = -1

    def next_frame(self):
        self.FrameIndex += 1
        # 必须 copy —— read_video_improved 内部会做 in-place 修改
        img = np.array(self._arr[self.FrameIndex], copy=True)
        if self.flag_rotate:
            img = np.rot90(img)
        if self.infiledatatype == 'uint8':
            img = np.asarray(img, dtype='uint16') * 256
        return img

    def close(self):
        # numpy buffer view 必须先释放,再 unmmap,再关 fd (顺序敏感)
        if hasattr(self, '_arr') and self._arr is not None:
            del self._arr
            self._arr = None
        if hasattr(self, '_mm') and self._mm is not None:
            try:
                self._mm.close()
            except (BufferError, OSError):
                pass
            self._mm = None
        if hasattr(self, '_fd') and self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

# wrapper of video_reader which stores everything in memory
class all_video_reader:
    def __init__(self, file, buffer_size = 25):
        vid_rdr = video_reader(file, buffer_size)
        self.file = file
        self.ih = vid_rdr.ih
        self.iw = vid_rdr.iw
        self.Width = vid_rdr.Width
        self.Height = vid_rdr.Height
        self.FrameCount = vid_rdr.FrameCount
        self.count = vid_rdr.count
        self.FrameIndex = -1
        self.frames = np.zeros((self.FrameCount, self.ih, self.iw), dtype=np.uint16)
        # load all frames
        i = 0
        self.means = np.zeros(self.FrameCount)
        while vid_rdr.has_frames():
            frame = vid_rdr.next_frame()
            self.means[i] = np.mean(frame)
            self.frames[i, :, :] = frame
            i+=1

    def has_frames(self):
        return self.FrameIndex + 1 < self.FrameCount

    def next_frame(self):
        self.FrameIndex += 1
        return self.frames[self.FrameIndex, :, :]

    def reset(self):
        self.FrameIndex = -1

    
    
    
