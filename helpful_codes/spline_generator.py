# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 09:36:49 2026

@author: nickjc
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits



'''
SETUP
'''



# characters that already do things on press in matplotlib
rcparam_chars = \
    plt.rcParams['keymap.back'] + plt.rcParams['keymap.copy'] + \
    plt.rcParams['keymap.forward'] + plt.rcParams['keymap.fullscreen'] + \
    plt.rcParams['keymap.grid'] + plt.rcParams['keymap.grid_minor'] + \
    plt.rcParams['keymap.help'] + plt.rcParams['keymap.home'] + \
    plt.rcParams['keymap.pan'] + plt.rcParams['keymap.quit'] + \
    plt.rcParams['keymap.save'] + plt.rcParams['keymap.xscale'] + \
    plt.rcParams['keymap.yscale'] + plt.rcParams['keymap.zoom']

# list of allowed characters 
chars = ['enter', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
allowed_chars_list = [char for char in chars if char not in rcparam_chars]
allowed_chars = ''
for char in allowed_chars_list:
    allowed_chars += char + ', '
allowed_chars = allowed_chars[:-2]

# function to load a fits cube
def fits_load(file_loc):
        #load in the data
        with fits.open(file_loc) as hdul:
            header = hdul[1].header
            data_cube = hdul[1].data

        #wavelength data from header
        number_wavelengths = header["NAXIS3"]
        wavelength_increment = header["CDELT3"]
        wavelength_start = header["CRVAL3"]
        ref_pix = header['CRPIX3'] # reference pixel value, usually 1
        
        wavelengths = (np.arange(number_wavelengths) + ref_pix - 1) * wavelength_increment + wavelength_start
        return wavelengths, data_cube
    
# function to load with numpy loadtxt, assume cols 1 and 2 are for data
def txt_load(file_loc):
        #load in the data
        data_file = np.loadtxt(file_loc, unpack=True)
        wavelengths = data_file[0]
        
        # make data 3d for consistency
        data = data_file[1]
        data_cube = data[:, np.newaxis, np.newaxis]
        return wavelengths, data_cube

# function to generate a test cube
def test_cube():
    wavelengths = np.arange(5, 28, 0.1)
    shape = np.array([5, 10])
    data_cube = ((np.sin(wavelengths))**2)[:, np.newaxis, np.newaxis]*np.ones(wavelengths.shape + shape)
    return wavelengths, data_cube
    
# function that turns button presses into lines
def on_key_press(event):
    # correct key pressed
    if event.key == line_char:
        coords = event.xdata, event.ydata
        # key pressed inside plot
        if None not in coords:
            wave = coords[0]
            # key press inside wavelength range
            if wave > wavelengths[0] and wave < wavelengths[-1] and wave not in wave_list_1:
                wave_list_1.append(wave)
                y1, y2 = np.nanmin(data), np.nanmax(data)
                line = plt.plot([wave, wave], [y1, y2], color='green', linestyle='dashed')
                fig.canvas.draw()
                artist_list.append(line[0])
                
                if len(wave_list_1) == 2 and num_cols == 2:
                    wave_1, wave_2 = wave_list_1
                    if wave_1 <= wave_2:
                        w1, w2 = wave_1, wave_2
                    else:
                        w1, w2 = wave_2, wave_1
                    wave_list_2.append([w1, w2])
                    # replace the 2 dotted lines with a solid box
                    line1 = artist_list[-1]
                    line1.remove()
                    artist_list.pop()
                    line2 = artist_list[-1]
                    line2.remove()
                    artist_list.pop()
                    box = plt.fill([w1, w2, w2, w1], [y1, y1, y2, y2], color='green', alpha=0.5)
                    fig.canvas.draw()
                    artist_list.append(box[0])
                    # cleaning up
                    wave_list_1.pop()
                    wave_list_1.pop()
    elif event.key == del_char:
        # artist_list will contain either a list of lines, a list of boxes, or a list of boxes and 1 line
        coords = event.xdata, event.ydata
        # key pressed inside plot, do nothing if artist list empty
        if None not in coords and len(artist_list) > 0:
            if num_cols == 1:
                wave_list_1.pop()
                line = artist_list[-1]
                line.remove()
                artist_list.pop()
            elif num_cols == 2 and len(wave_list_1) == 1:
                wave_list_1.pop()
                line = artist_list[-1]
                line.remove()
                artist_list.pop()
            else:
                wave_list_2.pop()
                box = artist_list[-1]
                box.remove()
                artist_list.pop()
            fig.canvas.draw()



'''
GATHERING INPUTS
'''



# get type of character to use in function to make lines
print('Type character to press for line selection:')
go_on = 0
while go_on == 0:
    line_char = str(input())
    if line_char not in allowed_chars_list:
        print(f'{line_char} is currently in rcParams, or is not a single character. The allowed characters are {allowed_chars}. \nPlease try again:')
    else:
        allowed_chars_list.remove(line_char)
        allowed_chars = ''
        for char in allowed_chars_list:
            allowed_chars += char + ', '
        allowed_chars = allowed_chars[:-2]
        go_on = 1

# get type of character to use in function to delete last line
print('Type character to press to delete the last line:')
go_on = 0
while go_on == 0:
    del_char = str(input())
    if del_char not in allowed_chars_list:
        print(f'{del_char} is currently in use, is currently in rcParams, or is not a single character. The allowed characters are {allowed_chars}. \nPlease try again:')
    else:
        allowed_chars_list.remove(del_char)
        allowed_chars = ''
        for char in allowed_chars_list:
            allowed_chars += char + ', '
        allowed_chars = allowed_chars[:-2]
        go_on = 1

# get file loc of data to plot
print('Enter path to file to plot:')
go_on = 0
while go_on == 0:
    file_loc = str(input())
    if '.fits' in file_loc:
        try:
            wavelengths, data_cube = fits_load(file_loc)
        except:
            print('Path to a .fits file specified, but it is incompatible with the JWST fits standard. \nPlease try again:')
        else:
            go_on = 1
    elif file_loc == 'test':
        wavelengths, data_cube = test_cube()
        go_on = 1
    else:
        try:
            wavelengths, data_cube = (file_loc)
        except:
            print('Path to a non-.fits file specified, but it is incompatible with np.loadtxt. \nPlease try again:')
        else:
            go_on = 1

# getting coordinates to plot
y_comp, x_comp = data_cube.shape
print(f'Data index to plot. max indices are {y_comp - 1}, {x_comp - 1}:')
go_on = 0
while go_on == 0:
    index = str(input())
    try: 
        y, x = [int(float(index.split(',')[i])) for i, line in enumerate(index.split(','))]
    except:
        print('Input does not consist of 2 numbers seperated by 1 comma. \nPlease try again:')
    else:
        if y >= data_cube.shape[0] or x >= data_cube.shape[1]:
            print(f'Indices of {y, x} are outside of range, max indices are {y_comp - 1}, {x_comp - 1}. \nPlease try again:')
        else:
            data = data_cube[:, y, x]
            go_on = 1
    
# format of coords
print('Save wave_list with 1 or 2 columns:')
go_on = 0
while go_on == 0:
    try:
        num_cols = int(input())
    except:
        print('Input is not an integer. \nPlease try again:')

    if num_cols == 1:
        go_on = 1
    elif num_cols == 2:
        go_on = 1
    else:
        print(f'{num_cols} is an integer, but is not 1 or 2. \nPlease try again:')



'''
INTERACTIVE PLOT
'''



wave_list_1 = []
wave_list_2 = []
artist_list = []
fig = plt.figure(f'Data Cube Indices: {y}, {x}. Key to press for line: {char}. Close when done placing lines')
plt.plot(wavelengths, data)

# checking for key presses
plt.gca().figure.canvas.mpl_connect('key_press_event', on_key_press)

# stops the CLI until the user closes the graph window (Command Line Interface)
plt.show(block=True) 



'''
LIST ORDERING
'''


wave_list = []
if num_cols == 1:
    while len(wave_list_1) > 0:
        wave = min(wave_list_1)
        wave_list.append(wave)
        wave_list_1.remove(wave)
else:
    # pairs have smaller first, so just need to make the first in each pair ordered
    while len(wave_list_2) > 0:
        wave_pair = min(wave_list_2)
        wave_list.append(wave_pair)
        wave_list_2.remove(wave_pair)
    


'''
SAVING LOGIC
'''



# should saving occur
print('Save list? [y/n]:')
go_on = 0
while go_on == 0:
    save = str(input())
    if save == 'y':
        proceed = True
        go_on = 1
    elif save == 'n':
        proceed = False
        go_on = 1
    else:
        print(f'{save} is not y or n. \nPlease try again:')

if proceed == True:
    # file loc
    print('Path to save file to. Leave blank to save here:')
    save_loc = str(input())
    # remove any extra /
    files = save_loc.split('/')
    short_files = [file for file in files if file != '']
    # rebuild with exact number of / needed
    save_loc = ''
    if len(short_files) > 0:
        for file in short_files:
            save_loc += file + '/'
    # making path if it does not exist
    if not os.path.exists(save_loc) and save_loc != '':
        os.makedirs(save_loc)
            
    # file name
    print('Name of the file to save info in. Leave blank to name it wave_list.txt:')
    go_on = 0
    while go_on == 0:
        file_loc = str(input())
        if file_loc == '':
            file_loc = 'wave_list.txt'
        # see if the file exists, ask to replace if it does
        try:
            f = open(save_loc + file_loc, 'r')
        except:
            proceed_name = True
        else:
            f.close()
            go_on_1 = 0
            print(f'{file_loc} already exists. Replace it? [y/n]:')
            while go_on_1 == 0:
                replace = str(input())
                if replace == 'y':
                    proceed_name = True
                    go_on_1 = 1
                elif replace == 'n':
                    proceed_name = False
                    go_on_1 = 1
                else:
                    print(f'{replace} is not y or n. \nPlease try again:')
        if proceed_name == True:
            go_on = 1

    # writing data to file
    with open(save_loc + file_loc, 'w') as file:
        if num_cols == 1:
            for wave in wave_list:
                line = '{:.4f}'.format(wave) + '\n'
                file.write(line)
        else:
            for wave_pair in wave_list:
                line = '{:.4f}'.format(wave_pair[0]) + ' ' + '{:.4f}'.format(wave_pair[1]) + '\n'
                file.write(line)

    

    


























