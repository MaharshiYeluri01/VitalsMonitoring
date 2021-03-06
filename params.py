#FFT
PASSBAND_HIGH=3
PASSBAND_LOW=0.8

#mediapipe
PRESENCE_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.5
MIN_DETECTION_CONFIDENCE=0.5
STATIC_IMAGE_MODE=True
MAX_FACES=1

#face segmentation
FOREHEAD_PAD=0.4

#video
SECONDS=30
EXTENSIONS=['.mkv']



# default box coords
BOX_HPAD=0.3
BOX_VPAD=0.15
BW=220*1280//600
BH=300*720//400
X=1280//3
Y=720//6

pw=int(BOX_HPAD*BW)
ph=int(BOX_VPAD*BH)

BOX=(X,Y,BW,BH)
extended_box=(X-pw,Y-ph,BW+2*pw,BH+2*ph)