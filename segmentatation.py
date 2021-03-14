import math
from capture import *
from vitals_api_videos.patch_cords import *
from typing import List, Tuple, Union
from params import *

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

face_mesh=mp_face_mesh.FaceMesh(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_faces=MAX_FACES,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE)


left_cheek_cords = lambda fc:fc[left_cheek]
right_cheek_cords = lambda fc:fc[right_cheek]
forehead_cords = lambda fc:fc[forehead]


def get_patch(img,patch_coords,is_face):
    r,c,_=img.shape
    if not is_face:
        polygons1=cv2.convexHull(patch_coords[0])
        polygons2=cv2.convexHull(patch_coords[1])
        polygons3=cv2.convexHull(patch_coords[2])
        polygons=[polygons1,polygons2,polygons3]
    else:polygons=[cv2.convexHull(patch_coords)]
    mask = np.zeros((r,c), dtype=np.uint8)
    mask=cv2.fillPoly(mask, polygons, (1))

    patch=img*mask.reshape(r,c,1)
    return patch.sum(axis=(0,1))/(mask[mask==1]).sum()

def get_landmarks(image,landmark_list):
  if not landmark_list:
    return
  if image.shape[2] != 3:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  cords=[]
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    cords.append(landmark_px)
  return cords

def _normalized_to_pixel_coordinates(normalized_x, normalized_y,image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px,y_px

# Todo:nb_frames == nb_cords
def get_annotated_cords(frames):
    annotated_cords=[]
    for image in frames:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        continue
    for face_landmarks in results.multi_face_landmarks:
        cords=get_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,)
    annotated_cords.append(cords)
    return annotated_cords




def overlay_mask(frames,annotated_cords):
    patch_signal=[]
    for img,cords in zip(frames,annotated_cords):
        cords=np.array(cords,dtype='int32')
        
        le=left_eye_cords(cords)
        re=right_eye_cords(cords)
        mt=lip_cords(cords)
        fc=face_cords(cords)
        patch_signal.append(get_patch_fullface(img,[le,re,mt,fc]))
    return np.array(patch_signal)


def update_upper_forehead(cords):
    updated_cords=[]
    for i in upper_forehead:
        x,y=cords[i]
        updated_cords.append([x,max(0,y-int(y*FOREHEAD_PAD))])
    return updated_cords
def face_cords(fc):
    uc=update_upper_forehead(fc)
    return np.array(list(fc)+uc,dtype='int32')

left_eye_cords=lambda fh:fh[lefteye]
right_eye_cords=lambda fh:fh[righteye]
lip_cords=lambda fh:fh[lips]

def get_patch_fullface(img,patch_cords):
    r,c,_=img.shape
    polygons=[cv2.convexHull(patch_cords[0]),cv2.convexHull(patch_cords[1]),cv2.convexHull(patch_cords[2])]
    mask = np.ones((r,c), dtype=np.uint8)
    mask=cv2.fillPoly(mask, polygons, (0))
    
    
    polygons=[cv2.convexHull(patch_cords[3])]
    mask1 = np.zeros((r,c), dtype=np.uint8)
    mask1=cv2.fillPoly(mask1, polygons, (1))
    
    mask=mask*mask1
    patch=img*mask.reshape(r,c,1)
    return patch.sum(axis=(0,1))/(mask[mask==1]).sum()

def masked_image(img,cords):
    cords=np.array(cords,dtype='int32')

    le=left_eye_cords(cords)
    re=right_eye_cords(cords)
    mt=lip_cords(cords)
    fc=face_cords(cords)
    r,c,_=img.shape
    pgns=[cv2.convexHull(le),cv2.convexHull(re),cv2.convexHull(mt)]
    mask = np.ones((r,c), dtype=np.uint8)
    mask=cv2.fillPoly(mask, pgns, (0))
    
    
    pgns=[cv2.convexHull(fc)]
    mask1 = np.zeros((r,c), dtype=np.uint8)
    mask1=cv2.fillPoly(mask1, pgns, (1))
    
    mask=mask*mask1
    patch=img*mask.reshape(r,c,1)
    return patch
