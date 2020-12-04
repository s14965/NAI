'''
Authors: Jerzy Rześniowiecki, Szymon Maj 
Fuzzy Logic Office Temperature Autonomous Control System

Input:
    - room temperature
    - humidity
    - fan speed
    
Output:
    - warmup/cool command
    - speedup/slowdown command
    
'''

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temp= ctrl.Antecedent(np.arange(0, 41), 'temperature')
hum = ctrl.Antecedent(np.arange(0, 101), 'humidity')
speed = ctrl.Antecedent(np.arange(0, 101), 'speed')

# Consequents
cmd = ctrl.Consequent(np.arange(15, 26), 'command')
cmd2 = ctrl.Consequent(np.arange(15, 26), 'command2')

# Temperature memberships
temp['coldest'] = fuzz.trapmf(temp.universe, [0, 4, 6, 8])
temp['cool'] = fuzz.trapmf(temp.universe, [6, 10, 12, 16])
temp['warm'] = fuzz.trapmf(temp.universe, [12, 16, 18, 24])
temp['hot'] = fuzz.trapmf(temp.universe, [18, 22, 24, 32])
temp['hottest'] = fuzz.trapmf(temp.universe, [24, 28, 30, 40])

# Humidity memberships
hum['low'] = fuzz.gaussmf(hum.universe, 0, 15)
hum['optimal'] = fuzz.gaussmf(hum.universe, 50, 15)
hum['high'] = fuzz.gaussmf(hum.universe, 100, 15)

# Fan speed memberships
speed['slow'] = fuzz.gaussmf(speed.universe, 0, 15)
speed['medium'] = fuzz.gaussmf(speed.universe, 50, 15)
speed['fast'] = fuzz.gaussmf(speed.universe, 100, 15)

# Command memberships
cmd['cool'] = fuzz.trimf(cmd.universe, [15, 17, 20])
cmd['warmup'] = fuzz.trimf(cmd.universe, [18, 20, 26])

cmd2['slowdown'] = fuzz.trimf(cmd2.universe, [15, 17, 20])
cmd2['speedup'] = fuzz.trimf(cmd2.universe, [18, 20, 26])

# Rule system
# Rules for warming up
rule1 = ctrl.Rule(
    (temp['coldest'] & speed['slow']) |
    (temp['cool'] & hum['low'] & speed['slow']) |
    (temp['cool'] & hum['optimal'] & speed['slow']) |
    (temp['warm'] & hum['low'] & speed['slow']), [cmd['warmup'], cmd2['speedup']])

# Rules for cooling down
rule2 = ctrl.Rule(
    (temp['coldest'] & speed['medium']) |
    (temp['coldest'] & speed['fast']) |
    (temp['cool'] & hum['low'] & speed['medium']) |
    (temp['cool'] & hum['optimal'] & speed['medium']) |
    (temp['cool'] & hum['low'] & speed['fast']) |
    (temp['cool'] & hum['optimal'] & speed['fast']), [cmd['warmup'], cmd2['slowdown']])

# Rules for speeding up fan
rule3 = ctrl.Rule(
    (temp['warm'] & hum['optimal'] & speed['slow']) |
    (temp['warm'] & hum['high'] & speed['slow']) |
    (temp['warm'] & hum['low'] & speed['medium']) |
    (temp['warm'] & hum['high'] & speed['medium']) |
    (temp['hot'] & hum['optimal'] & speed['slow']) |
    (temp['hot'] & hum['high'] & speed['slow']) |
    (temp['hot'] & hum['optimal'] & speed['medium']) |
    (temp['hot'] & hum['high'] & speed['medium']) |
    (temp['hottest'] & speed['slow']) |
    (temp['hottest'] & speed['medium']), [cmd['cool'], cmd2['speedup']])

# Rules for slowing down fan
rule4 = ctrl.Rule(
    (temp['warm'] & speed['fast']) |
    (temp['hot'] & hum['optimal'] & speed['fast']) |
    (temp['hot'] & hum['high'] & speed['fast']) |
    (temp['hottest'] & speed['fast']), [cmd['cool'], cmd2['slowdown']])

# Control System Creation and Simulation
cmd_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
cmd_output = ctrl.ControlSystemSimulation(cmd_ctrl)

cmd2_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
cmd2_output = ctrl.ControlSystemSimulation(cmd2_ctrl)


# Enter values to test
# Temperature
temperature_value = float(input("Enter temperature(1-39): "))

while temperature_value < 0 or temperature_value > 40:
    try:
        temperature_value = float(input("Please choose a number between 1 and 39 "))
    except ValueError:
        print('We expect you to enter a valid integer')

# Humidity
humidity_value = float(input("Enter humidity(1-99): "))

while humidity_value < 0 or humidity_value > 100:
    try:
        humidity_value = float(input("Please choose a number between 1 and 99 "))
    except ValueError:
        print('We expect you to enter a valid integer')
    
# Speed
speed_value = float(input("Enter speed(1-99): "))

while speed_value < 0 or speed_value > 100:
    try:
        speed_value = float(input("Please choose a number between 1 and 99 "))
    except ValueError:
        print('We expect you to enter a valid integer')

cmd_output.input['temperature'] = temperature_value
cmd_output.input['humidity'] = humidity_value
cmd_output.input['speed'] = speed_value
cmd_output.compute()

cmd2_output.input['temperature'] = temperature_value
cmd2_output.input['humidity'] = humidity_value
cmd2_output.input['speed'] = speed_value
cmd2_output.compute()

# Print output command and plots
print("Command and Command2 are defined between 15 and 25")
print("Command value:",cmd_output.output['command'])
print("Command2 value:",cmd2_output.output['command2'])

if (cmd_output.output['command'] > 20):
    print('Warm Up')
elif (cmd_output.output['command'] < 20 and cmd_output.output['command'] > 18):
    print('No change')
else:
    print('Cool Down')


if (cmd2_output.output['command2'] > 20):
    print('Speed Up')
elif (cmd2_output.output['command2'] < 20 and cmd2_output.output['command2'] > 18):
    print('No change')
else:
    print('Slow Down')
