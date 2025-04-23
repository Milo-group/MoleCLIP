import sys
sys.path.append('..')

from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Geometry import Point2D
from rdkit.Chem import AllChem

from PIL import Image
import cv2
import io
import numpy as np
from math import floor
import random
from cairosvg import svg2png
import time
import multiprocessing as mp
import concurrent.futures


import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2


fontlist = ['DejaVuSerif-BoldItalic.ttf', 'DejaVuSerif-Bold.ttf', 'DejaVuSansMono-BoldOblique.ttf', 'DejaVuSans.ttf', 'cmmi10.ttf', 
                'cmtt10.ttf', 'STIXGeneral.ttf', 'DejaVuSerif-Italic.ttf', 'STIXGeneralItalic.ttf', 'cmb10.ttf', 'DejaVuSansMono-Bold.ttf',
                'DejaVuSansMono-Oblique.ttf', 'cmss10.ttf', 'cmr10.ttf', 'DejaVuSerif.ttf', 'STIXGeneralBol.ttf', 'STIXGeneralBolIta.ttf', 
                'DejaVuSans-Bold.ttf', 'DejaVuSans-BoldOblique.ttf', 'DejaVuSans-Oblique.ttf', 'DejaVuSansMono.ttf']


def calc_mask(dm, img_str, R_mask):
    img_str_list = img_str.split("<")
    
    
    atoms = len(list(dm.GetAtoms()))
    explicit = []
    for i in range(atoms):
        for line in img_str_list:
            if f"'atom-{i}'" in line and 'bond' not in line and "width" not in line:
                explicit.append(i)
                break
    mask = [i for i in range(atoms) if i not in explicit]

    if R_mask == []:
        
        return  mask, img_str
    
    else:
        new_str_list =[]
            
        for idx, line in enumerate(img_str_list):
            flag = 0
            for i in R_mask:
                if f"'atom-{i}'" in line:
                    flag = 1

            if idx >= len(img_str_list) - len(R_mask)*2 - 1:
                if "#" in line:
                    line = line[:-17] + "fill='#000000'/>"
            if flag == 0:
                new_str_list.append(line)

        final_str = "<".join(new_str_list).replace("#33CCCC", "#000000")
        final_str = final_str.replace("#191919", "#000000")
        return  mask, final_str


def R_group_replacement(mol):
    
    R_masks = []
    mw = Chem.RWMol(mol)
    symbols = ['O', 'N', 'S', 'F', 'Cl', 'Br', 'I']
    atoms = len(list(mol.GetAtoms()))
    
    for symbol in symbols:
        symbol_R_mask = []
        for idx in range(atoms):
            atom = list(mol.GetAtoms())[idx]
            if len(atom.GetBonds()) == 1 and atom.GetSymbol() == symbol:

                bond = list(atom.GetBonds())[0]

                if str(bond.GetBondType()) != "SINGLE":
                    continue

                else:
                    mw.ReplaceAtom(idx, Chem.rdchem.Atom('Xe'))
                    if bond.GetBeginAtomIdx() == idx:
                        symbol_R_mask.append([idx, bond.GetEndAtomIdx()])
                    else:
                        symbol_R_mask.append([idx, bond.GetBeginAtomIdx()])           
        R_masks.append(symbol_R_mask)
                 
                
    mol = mw.GetMol()

    return mol, R_masks

def mol_to_img(mol, rand = False, r_replace = False, filename = None):

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)  

    if r_replace:
        mol, R_masks = R_group_replacement(mol)

    drawer = Draw.MolDraw2DSVG(224, 224)   
    dm = Draw.PrepareMolForDrawing(mol)
    
    #bond width randomization 
    if rand:   
        width = random.choice(range(2,6))
        drawer.drawOptions().bondLineWidth = width
        
        #font size randomization
        fontsize = random.choice(range(14, 25, 1))
        drawer.drawOptions().minFontSize = fontsize
        drawer.drawOptions().maxFontSize = fontsize
        
        #bond length randomization
        drawer.drawOptions().fixedBondLength = 20 + 80*random.random()
        
        #gap in multiple bonds randomization
        drawer.drawOptions().multipleBondOffset = 0.1 + 0.3*random.random()

        #rotation randomization
        drawer.drawOptions().rotate = 360*random.random()

        #font type randomization
        drawer.drawOptions().fontFile = f"fonts/{random.choice(fontlist)}" 
     
    drawer.DrawMolecule(dm)
    
    if r_replace:
        for mask_idx, mask in enumerate(R_masks):
            for idx1,idx2 in mask:
                if list(drawer.GetDrawCoords(idx1))[0] < list(drawer.GetDrawCoords(idx2))[0] + 5 and list(drawer.GetDrawCoords(idx1))[0] > list(drawer.GetDrawCoords(idx2))[0] - 5:
                    drawer.DrawString(f'R{mask_idx+1}', drawer.GetDrawCoords(idx1), 0 , rawCoords=True)
                elif list(drawer.GetDrawCoords(idx1))[0] > list(drawer.GetDrawCoords(idx2))[0]:
                    drawer.DrawString(f'R{mask_idx+1}', drawer.GetDrawCoords(idx1), 1 , rawCoords=True)
                else:
                    drawer.DrawString(f'R{mask_idx+1}', drawer.GetDrawCoords(idx1), 2 , rawCoords=True)

    drawer.FinishDrawing()
    img_str = drawer.GetDrawingText()
    
    if r_replace:
        R_mask = [item[0] for sublist in R_masks for item in sublist]
        mask, img_str = calc_mask(dm, img_str, R_mask)
    
    if filename != None:
        svg2png(bytestring=img_str, write_to=filename)
    
    else:
        img_bytes = svg2png(bytestring=img_str)
        pil_img = Image.open(io.BytesIO(img_bytes))
        cv2_img = np.array(pil_img)[:, :, ::-1].copy()  
        
        return cv2_img
                       




