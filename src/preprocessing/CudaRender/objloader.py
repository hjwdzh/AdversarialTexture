import numpy as np
import skimage.io as sio

def LoadTextureOBJ(model_path):
  vertices = []
  vertex_textures = []
  vertex_normals = []
  faces = []
  face_mat = []
  face_textures = []
  face_normals = []
  lines = [l.strip() for l in open(model_path)]
  materials = {}
  kdmap = []
  mat_idx = -1
  filename = model_path.split('/')[-1]
  file_dir = model_path[:-len(filename)]

  for l in lines:
    words = [w for w in l.split(' ') if w != '']
    if len(words) == 0:
      continue

    if words[0] == 'mtllib':
      model_file = model_path.split('/')[-1]
      mtl_file = model_path[:-len(model_file)] + words[1]
      mt_lines = [l.strip() for l in open(mtl_file) if l != '']
      for mt_l in mt_lines:
        mt_words = [w for w in mt_l.split(' ') if w != '']
        if (len(mt_words) == 0):
          continue
        if mt_words[0] == 'newmtl':
          key = mt_words[1]
          materials[key] = np.array([[[0,0,0]]]).astype('uint8')
        if mt_words[0] == 'Kd':
          materials[key] = np.array([[[float(mt_words[1])*255, float(mt_words[2])*255, float(mt_words[3])*255]]]).astype('uint8')
        if mt_words[0] == 'map_Kd':
          if mt_words[1][0] != '/':
            print(file_dir + mt_words[1])
            img = sio.imread(file_dir + mt_words[1])
          else:
            print(mt_words[1])
            img = sio.imread(mt_words[1])
          if len(img.shape) == 2:
            img = np.dstack((img,img,img))
          elif img.shape[2] >= 4:
            img = img[:,:,0:3]
          materials[key] = img

    if words[0] == 'v':
      vertices.append([float(words[1]), float(words[2]), float(words[3])])
    if words[0] == 'vt':
      vertex_textures.append([float(words[1]), float(words[2])])
    if words[0] == 'vn':
      vertex_normals.append([float(words[1]), float(words[2]), float(words[3])])
    if words[0] == 'usemtl':
      mat_idx = len(kdmap)

      kdmap.append(materials[words[1]])

    if words[0] == 'f':
      f = []
      ft = []
      fn = []
      for j in range(3):
        w = words[j + 1].split('/')[0]
        wt = words[j + 1].split('/')[1]
        wn = words[j + 1].split('/')[2]
        f.append(int(w) - 1)
        ft.append(int(wt) - 1)
        fn.append(int(wn) - 1)
      faces.append(f)
      face_textures.append(ft)
      face_normals.append(fn)
      face_mat.append(mat_idx)
  F = np.array(faces, dtype='int32')
  V = np.array(vertices, dtype='float32')
  V = (V * 0.5).astype('float32')
  VN = np.array(vertex_normals, dtype='float32')
  VT = np.array(vertex_textures, dtype='float32')
  FT = np.array(face_textures, dtype='int32')
  FN = np.array(face_normals, dtype='int32')
  face_mat = np.array(face_mat, dtype='int32')

  print(VN.shape, FN.shape)

  return V, F, VT, FT, VN, FN, face_mat, kdmap
