# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:27:15 2024

@author: nickj
"""

'''
class tutorial
'''


import random


#%%

#example 1: storing objects with info

class example1():
    '''
    Stores planets, and whether or not they are cool
    '''

    mercury = 'cool'
    mars = 'not cool'
    saturn = 'cool'
    pluto = 'not a planet'
    venus = 7
    
#%%

#example 2: __init__

class example2():
    
    '''
    Stores planets, and whether or not they are cool, the useful way
    
    Parameters
    ----------
    planet
        TYPE: string
        DESCRIPTION: name of the planet
    coolness
        TYPE: string
        DESCRIPTION: if the planet is cool or not
    '''
    
    def __init__(self, planet, coolness):
        self.planet = planet
        self.coolness = coolness
        
creative_name = example2('mercury', 'cool')
im_so_creative = example2('mars', 'not cool')

#%%

#example 2.5: why self? also shenanagans

class example2_point_5():
    
    '''
    Stores planets, and whether or not they are cool, the useful way
    
    Parameters
    ----------
    planet
        TYPE: string
        DESCRIPTION: name of the planet
    coolness
        TYPE: string
        DESCRIPTION: if the planet is cool or not
    '''
    
    def __init__(teflon, planet, coolness):
        teflon.planet = coolness+' hello' #what happens when a string is added?
        teflon.coolness = planet #what happens when coolness and planet are swapped?

creative_teflon_name = example2_point_5('mercury', 'cool')
im_so_teflon_creative = example2_point_5('mars', 'not cool')

#%%

#example 2.5.5: remove self

class example2_point_5_point_5():
    
    '''
    Stores planets, and whether or not they are cool, the useful way
    
    Parameters
    ----------
    planet
        TYPE: string
        DESCRIPTION: name of the planet
    coolness
        TYPE: string
        DESCRIPTION: if the planet is cool or not
    '''
    
    def __init__(teflon, planet, coolness):
        pass
        

creative_teflon_name_again = example2_point_5_point_5('mercury', 'cool')


#%%

#example 3: methods

class example3():
    
    '''
    Stores planets, and whether or not they are cool, the useful way
    
    Parameters
    ----------
    planet
        TYPE: string
        DESCRIPTION: name of the planet
    coolness
        TYPE: string
        DESCRIPTION: if the planet is cool or not
    '''
    
    def __init__(self, planet, coolness=None):
        self.planet = planet
        self.coolness = coolness
        
    def coolness_finder(self):
        '''
        Determines coolness
        '''
        
        random.seed(10)
        coolness_value = random.randint(0, 10)
        
        if coolness_value > 5:
            coolness = 'cool'
            
        else:
            coolness = 'not cool'
            
        self.coolness = coolness
        self.coolness_value = coolness_value
        
planet_creative_name = example3('mercury')

planet_creative_name.coolness_finder()

#%%

#example 4: using classes to call classes

class dummy_class():

    def number_generator(self):
        random.seed(5)
        coolness_value = random.randint(0, 10)
        self.coolness_value = coolness_value



class example4():
    
    '''
    Stores planets, and whether or not they are cool, the useful way
    
    Parameters
    ----------
    planet
        TYPE: string
        DESCRIPTION: name of the planet
    coolness
        TYPE: string
        DESCRIPTION: if the planet is cool or not
    '''
    
    def __init__(self, planet, coolness=None):
        self.planet = planet
        self.coolness = coolness

    def coolness_finder(self):
        '''
        Determines coolness
        '''
        
        number = dummy_class()
        number.number_generator()
        coolness_value = number.coolness_value
        
        if coolness_value > 5:
            coolness = 'cool'
            
        else:
            coolness = 'not cool'
            
        self.coolness = coolness
        self.coolness_value = coolness_value
        
     
        
planet_creative_name = example4('mercury')
planet_creative_name.coolness_finder()

#%%

#example 5: multiple instances

class example5():
    
    '''
    Stores planets, and whether or not they are cool, the useful way
    
    Parameters
    ----------
    planet
        TYPE: string
        DESCRIPTION: name of the planet
    coolness
        TYPE: string
        DESCRIPTION: if the planet is cool or not
    '''
    
    def __init__(self, planet, coolness=None):
        self.planet = planet
        self.coolness = coolness
     
        
     
    def coolness_finder(self):
        '''
        Determines coolness
        '''
        
        coolness_value = random.randint(0, 10)
        
        if coolness_value > 5:
            coolness = 'cool'
            
        else:
            coolness = 'not cool'
            
        self.coolness = coolness
        self.coolness_value = coolness_value        
        
        
        
    def coolness_comparison(self, other_planet):
        '''
        compares coolness between 2 planets
        
        Parameters
        ----------
        other_planet
            TYPE: example5 instance
            DESCRIPTION: another planet to compare to
        '''
        coolness_value = self.coolness_value
        other_coolness_value = other_planet.coolness_value
        
        if coolness_value > other_coolness_value:
            print(self.planet + ' is cooler than ' + other_planet.planet)
            
        elif coolness_value < other_coolness_value:
            print(self.planet + ' is not cooler than ' + other_planet.planet)
            
        else:
            print('ERROR')
            
            
        
planet_creative_name = example5('mercury')
planet_creative_name.coolness_finder()

other_planet_creative_name = example5('neptune')
other_planet_creative_name.coolness_finder()

planet_creative_name.coolness_comparison(other_planet_creative_name)

#%%


