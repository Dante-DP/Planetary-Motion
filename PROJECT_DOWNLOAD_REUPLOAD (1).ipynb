# Planetary Motion
## by Aahan Arakkal, Tess Bentley, Toby Clifft, Dante du Preez and Diana Yuan (the Elliptical Explorers)

## Introduction

In this project, we explore the motion of planets through a vast range of numerical techniques. We start by verifying the accuracy of the velocity Verlet algorithm in solving Newton's equations of motion for a one-dimensional harmonic oscillator. 
Following this, we estimate how the Earth moves around the sun by simulating an earth-like planet's orbit given certain assumptions, which are stated later on.
Our fourth section looks at how varying starting conditions can lead to different orbital shapes: ellipses, hyperbolae and, in a specific case, a parabola. For bounded elliptical trajectories we compute perihelion, aphelion, eccentricity, and
other orbital parameters to further explore what determines the shape of an orbit. We also examined how escape velocity is determined from energy considerations, comparing our numerical results with theoretical predictions.
Finally, our extension looks at the prediction of solar eclipses. We extend our simulation to a three-body problem involving the Sun, Earth and Moon; modelling their motion from given starting conditions to figure out when they will align
to form an eclipse, then identifying which type of eclipse occurs.

## Constants
The constants which we will use throughout the project are listed below


```python
# Gravitational constant
G = 6.67348e-11              # (in m^3 / kgs^2 )

# Sun values
m_sun = 1.988420392e30       # Mass of sun (in kg)
r_sun = 696340e3             # Radius of sun (in m)

# Earth values
m_earth = 5.972000000e24     # Mass of Earth (in kg)
r_earth = 6371e3             # Radius of Earth (in m)

# Moon values
m_moon = 7.345828157e22      # Mass of the Moon (in kg)
r_moon = 1737e3              # Radius of the Moon (in m)

# Astronomical Constants
R = 149.6e9                  # 1AU in meters

# Time Values
hr1 = 3600                   # Time in 1 Hour
yr1 = 365.25*24*3600         # Time in 1 year
```

## Libraries
Below is a list of all the libraries which must be imported for this project:


```python
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lag


# Specific to Q9:
import pandas as pd
import time
from datetime import datetime, timedelta
from tqdm import tqdm
```

## Q1
This first question compares numerical and analytical solutions to differential equations describing a one-dimensional harmonic oscillator. We go on to plot the particle's motion over time, as well as its phase-space trajectory which shows how velocity varies with displacement.


```python
# Parameters
m = 1.0       
k = 1.0   
omega = np.sqrt(k/m)

# Initial conditions
x0 = 1.0      
v0 = 0.0      

# Max time and step size
t_max = 20.0
dt = 0.01
t = np.arange(0, t_max + dt, dt)
N = len(t)

# Exact solution
x_exact = x0 * np.cos(omega*t) + (v0/omega) * np.sin(omega*t)
v_exact = -x0 * omega * np.sin(omega*t) + v0 * np.cos(omega*t)

# Plot x(t) 
plt.figure()
plt.plot(t, x_exact, label='Exact x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Harmonic Oscillator: Position vs Time')
plt.legend()
plt.grid(True)

# Phase-space (x(t), v(t)) comparison 
plt.figure()
plt.plot(x_exact, v_exact, label='Exact (x,v)')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Harmonic Oscillator: Phase-Space Trajectory')
plt.legend()
plt.grid(True)

plt.show()

# Create arrays to store the numerical solution
x_verlet_alg = np.zeros(N)
v_verlet_alg = np.zeros(N)

# Set the values we have initially to the first index in the arrays
x_verlet_alg[0] = x0
v_verlet_alg[0] = v0

# The acceleration function
def acceleration(x):
    return -(k/m) * x

# Create a loop to use the verlet algorithm
for i in range(N - 1):
    current_a = acceleration(x_verlet_alg[i]) #the acceleration currently
    updated_x = x_verlet_alg[i] + v_verlet_alg[i]*dt + 0.5*current_a*(dt**2) #the updated position
    updated_a = acceleration(updated_x) #the updated acceleration, based on position
    updated_v = v_verlet_alg[i] + 0.5*(current_a + updated_a)*dt #the updated velocity, based on acceleration

    x_verlet_alg[i+1] = updated_x #store the results in the arrays
    v_verlet_alg[i+1] = updated_v

# Plot for numerical x(t)
plt.figure() 
plt.plot(t, x_verlet_alg, 'orange', linestyle=':', label='Numerical x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Harmonic Oscillator: Numerical x(t) with Verlet Algorithm')
plt.legend()
plt.grid(True)

# Plot for numerical (x,v) - the Phase-space
plt.figure() 
plt.plot(x_verlet_alg, v_verlet_alg, 'orange', linestyle=':', label='Numerical (x,v)')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Verlet Solution : (x,v) plot')
plt.legend()
plt.grid(True)

# The comparison plots for x(t)
plt.figure()
plt.plot(t, x_exact, label='Exact x(t)')
plt.plot(t, x_verlet_alg, color='orange', linestyle=':', label='Numerical x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Exact vs Numerical: x(t) over t')
plt.legend()
plt.grid(True)

# The comparison plots for (x,v) - the Phase-space
plt.figure() 
plt.plot(x_exact, v_exact, label='Exact (x,v)')
plt.plot(x_verlet_alg, v_verlet_alg, color='orange', linestyle=':', label='Numerical (x,v)')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Exact vs Numerical: (x,v)')
plt.legend()
plt.grid(True)

plt.show
```


    
![png](output_7_0.png)
    



    
![png](output_7_1.png)
    





    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_7_3.png)
    



    
![png](output_7_4.png)
    



    
![png](output_7_5.png)
    



    
![png](output_7_6.png)
    


### Graph interpretations q.1:
We have simulated the motion of an orbiting planet, using both the exact solution and the numerical solution, using the Verlet Algorithm.
Our first two graphs show the exact position x(t), giving us a sinusoidal curve and a phase-space trajectory (x,v), from which we get an elliptical curve. 
These perfect, smooth shapes demonstrate conservation of energy within our system. Using the Verlet algorithm, for the next two graphs of our numerical solutions, we find that these curves closely follow the previous two respectively.
The purpose of using 0.01 as a time step ensures any deviations in the two different solutions are minimised, which allows the numerical solution to mimic the exact orbit over time closely.

## Q2
We now go on to look at the orbit of an Earth-like planet around the sun. We apply Newton's law of gravitation in a simplified two-body system where the Sun is treated as fixed due to its much greater mass. We restrict the orbital motion to the xy-plane, since angular momentum is conserved. This setup provides a simple, yet fairly accurate, framework for simulating the planet's orbital trajectory.


```python
# Setup
s0 = np.sqrt(G * m_sun / R) # Orbital Speed of the Earth-Like Planet
r0 = np.array([R,0]) # Initial position vector of the Planet, Starting on the +ve x axis.
v0 = np.array([0, s0]) #Initial velocity vector of the planet (Perpendicular to the radius)

t = np.arange(0, yr1+hr1, hr1) # Time Array starting at 0, ending at 1 year, incrementing by an hour.
N = len(t) # Number of Increments of array t

# Results Storing arrays
r = np.zeros((N, 2))
v = np.zeros((N,2)) # Both Matrices initialised at 0 to store data.
r[0] = r0
v[0] = v0

# Gravitational Acceleration
def gAcceleration(x):
    dist = lag.norm(x)
    return -G * m_sun * x / dist**3 # By formula

# Verlet Integration Loop (variable_np1 = variable_{n+1})
a_n = gAcceleration(r[0]) # a_{n} is the initial gravitational acceleration
for i in range(N-1):
    # Updating Position, Acceleration, Velocity
    r_np1 = r[i] + v[i]*hr1 + 0.5*a_n*(hr1**2) # r_{n+1}  is updated
    a_np1 = gAcceleration(r_np1) # a_{n+1} is updated
    v_np1 = v[i] + 0.5*(a_n + a_np1)*hr1 # v_{n+1} is updated

    #Storing Variables
    r[i+1] = r_np1
    v[i+1] = v_np1
    a_n = a_np1

# Orbit Plot
plt.figure(figsize=(6,6))
plt.plot(r[:,0],r[:,1], label='Planet Trajectory')
plt.scatter(0,0,color='yellow',s=100,label='Sun')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.title('Simulation of an Earth-Like planets orbit')
plt.legend(loc='upper right')
plt.axis('equal')
plt.grid(True)
plt.show()

# Radius Over Time
plt.figure()
plt.plot(t/(24*hr1), lag.norm(r, axis=1))
plt.xlabel('Time (Days)')
plt.ylabel('Radius deviation from initial value (m)')
plt.title('Numerical deviation of orbital radius over time')
plt.grid(True)
plt.show() 
```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    


### Interpretation of Q2
This is a simulation of an earth-like planet around the sun using 2D Verlet integration. We start the planet initially at R = 149.6 x 10^11 m in the x direction.
We have the initial velocity perpendicular to R in order to keep the orbit in a circular trajectory. Our first plot shows the trajectory of the planet around the sun and we see we have an almost perfect circular orbit.
Consequently, we can see that the integration here conserves the angular momentum. Our second plot is the distance of the planet from the sun against time.
We can see this varies, due to numerical errors; however, the effect is minimal since our maximum deviation is clearly much less than the distance between the Earth and the Sun.

# Q3

In this section we will confirm that our simulated orbit generates a closed loop, given the initial conditions of an Earth like planet`.

We will also examine the total energy of the Earth throughout it's orbit, which is expected to remain constant.

Is the orbit a closed loop?
First off, let's check we have the correct initial conditions.


```python
print('')
print(f'r0 = {[r0[0] , r0[1]]}m = [R,0]m')
print(f'v0 = {[v0[0],v0[1]]}m/s ≈ [0,29800]m/s ')
```

    
    r0 = [np.float64(149600000000.0), np.float64(0.0)]m = [R,0]m
    v0 = [np.float64(0.0), np.float64(29782.72894968018)]m/s ≈ [0,29800]m/s 


So, $\mathbf{r}_0$ = R$\mathbf{\hat{x}}$, and $\mathbf{v}_0$ ≈ 29.8$\frac{\text{km}}{\text{s}}$$\mathbf{\hat{y}}$

Now we know our initial conditions are correct, we can check to see if our orbit is closed.

To do this we can take a closer look at the start and end points of our plot.


```python
start_r=np.array([r[0][0],0])
end_r=np.array([r[len(t)-1][0],r[len(t)-1][1]])
#Defines the start and end position for simulated orbit

plt.figure()
plt.grid(True)
plt.plot(r[:,0],r[:,1], label='Planet Trajectory')
plt.scatter(start_r[0],0,color='green',s=70,label='Start')
plt.scatter(end_r[0], end_r[1],color='red',s=70,label='End')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.title('Endpoints of Simulated Orbit')
plt.legend(loc='upper right')
plt.xlim(R-1e11,R+1e11)
plt.ylim(-.5e9,.5e9)
plt.show()
#Same plot from Q2. Have adjusted x and y axis limits to zoom in on start and end point.

print(f'Our Starting Position = {start_r} and our Ending Position = {end_r} (in metres)')
Close_error=lag.norm(start_r-end_r) #distance between start and end in m
Orbit_length= 2 * np.pi * R #circumfernece of expected orbit
Percentage_close_error=Close_error/Orbit_length*100 #computes percentage difference between expected vs total distance travelled in orbit
print(f'The distance error between start and end = {Close_error:.2} m')
print(f'Compared to the total length of orbit this generates a percentage error = {Percentage_close_error:.5f} %')
```


    
![png](output_16_0.png)
    


    Our Starting Position = [1.496e+11 0.000e+00] and our Ending Position = [ 1.49599971e+11 -9.32357924e+07] (in metres)
    The distance error between start and end = 9.3e+07 m
    Compared to the total length of orbit this generates a percentage error = 0.00992 %


Overall, the start and endpoints are very close compared to the scale of the orbit, so it is fair to say our simulation generates a closed loop. The error presented can be mainly attributed to the length of our simulation, which was approximated to 365.25 days.

### Is Energy Conserved?

We now look at the energy of the system and how it varies. Given that we are assuming the orbit is circular, we would expect that both the kinetic and potential energy remian constant.


```python
def System_energy(r,v,m_sun=m_sun,m_earth=m_earth): #Inputs are the position vand velocity vectors. Mass inputs are optional.
    Kinetic = 0.5 * m_earth * np.dot(v,v)
    Potential = -G  * m_earth * m_sun / lag.norm(r)
    Total = Kinetic + Potential
    return Kinetic, Potential, Total
#Have defined this as a function so we can reuse in Q4 if wanted. Returns kinetic, potential and total energy of Earth at any given point.

Energy=np.zeros((N,3)) #stores kinetic pootential and total energy at each point in simulatted orbit using the for loop.
for i in range(N):
    r_current=r[i]
    v_current=v[i]
    Energy[i] = System_energy(r_current,v_current,m_sun,m_earth)



t_days=t/(24*hr1) #to be used as x axis in many plots

plt.figure()
plt.plot(t_days,Energy[:,0], label=('Kinetic'))
plt.plot(t_days,Energy[:,1] , label='Potential')
plt.plot(t_days,Energy[:,2],label='Total')
plt.xlabel('Time (Days)')
plt.ylabel('Energy (J)')
plt.title('Conservation of Energies')
plt.legend()
plt.grid(True)
plt.show()
#plots all three energies against time in days.

fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(12,6))
ax1=ax[0]


ax1.plot(t_days,Energy[:,2],label='Total Energy')
max_en=max(Energy[:,2])
min_en=min(Energy[:,2])
en_range=abs(max_en-min_en)
ax1.set_title('Close up of Total Energy over Time')
ax1.set_ylabel('Energy (J)')
ax1.set_xlabel('Time (Days)')
ax1.set_ylim(min_en-en_range*0.2,max_en+en_range*0.2)
ax1.grid(True)
ax1.legend(loc='upper right')
#this is a graph of the total energy but heavily zoomed in on the y axis

ax2=ax[1] 
T_Energy_Change_Percentage=np.array([(Energy[i][2]-Energy[0][2])/Energy[0][2]*100 for i in range(len(t))]) #calculates percentage error against initial energy
ax2.plot(t_days,T_Energy_Change_Percentage,label='Total energy percentage change')
ax2.plot(t_days, [(max_en - Energy[0][2])/Energy[0][2]*100 for i in range(len(t))],linestyle='--',color='r', label='Max absolute energy percentage error')
ax2.set_title('Percentage Change of Total Energy')
ax2.set_xlabel('Time (Days)')
ax2.set_ylabel('Percentage Change from Starting Value (%)')
ax2.grid(True)
ax2.legend()
plt.show()
#graph shows percentage error of total energy only


print(f'Our maximum absloute energy percentage error is {max(abs(T_Energy_Change_Percentage))} % ~ {max(abs(T_Energy_Change_Percentage)):.12f} %')
```


    
![png](output_19_0.png)
    



    
![png](output_19_1.png)
    


    Our maximum absloute energy percentage error is 6.834088693098004e-12 % ~ 0.000000000007 %


We have a very small error margin here, which is most likely cause by the fact our verlet algorithm is a numerical aproximation.

As a result, we can conclude total energy does remain constant for our simulation.

### Analytical derivation of initial speed.

Since we are assuming the orbit is circular, we can use the circular motion equation:
$$ F = \frac{m v^2 }{R}    $$
We can also consider Newton's equation of motion for the Earth and the Sun:
$$ F = \frac{GMm}{R^2} $$
These forces must be equal, allowing us to iset the equations equal to each other:
$$ \frac{m v^2 }{R} = \frac{GMm}{R^2} $$
Now rearrange for v:
$$ mv^2 = \frac{GMm}{R} $$
$$ v^2 = \frac{GM}{R^2} $$
$$ v = \sqrt{\frac{GM}{R}} $$
$$ v = \sqrt{\frac{6.67348e-11 \times 1.988420392e30 }{149.6e9}} ≈ 29.7827 km/s  $$

### Q4

In this part we keep the initial position of the planet fixed and vary only the magnitude of
the tangential initial velocity. Using the velocity Verlet method, we simulate the resulting
orbits for several different choices of the initial speed. By doing this we can observe how
the orbit changes from a bounded ellipse to an unbounded hyperbolic trajectory as the total
energy increases.

For the cases that remain elliptical, we compute the perihelion, aphelion, semi–major axis,
semi–minor axis, and eccentricity. As we increase the initial velocity, the trajectory no
longer returns, allowing us to identify the escape velocity numerically. We then compare
this numerical value to the theoretical escape speed.

#### Import required libraries and define constants


```python
# Physical constants
G = 6.67430e-11
M = 1.989e30                 # mass of the Sun (kg)
R = 1.496e11                 # 1 AU (m)

# Circular velocity at radius R
v_circ = np.sqrt(G*M/R)

```

#### Define the gravitational acceleration 


```python
def acceleration(r):
    x, y = r
    dist = np.sqrt(x**2 + y**2)
    factor = -G*M / (dist**3)
    return np.array([factor*x, factor*y])

```

#### Implement the velocity Verlet integrator


```python
def verlet(r0, v0, dt, steps):
    r = np.zeros((steps, 2))
    v = np.zeros((steps, 2))
    r[0] = r0
    v[0] = v0

    a = acceleration(r0)

    for i in range(1, steps):
        r[i] = r[i-1] + v[i-1]*dt + 0.5*a*(dt**2)
        a_new = acceleration(r[i])
        v[i] = v[i-1] + 0.5*(a + a_new)*dt
        a = a_new

    return r, v
```

#### Function to compute orbital parameters for elliptical trajectories


```python
def orbital_parameters(r):
    dist = np.sqrt(r[:,0]**2 + r[:,1]**2)
    P = np.min(dist)
    A = np.max(dist)
    a = (A + P) / 2
    b = np.sqrt(A * P)
    e = (A - P) / (A + P)
    return P, A, a, b, e

```

#### Simulate trajectories for several choices of $\lambda$



```python
lambdas = [0.6, 0.8, 1.0, 1.2, 1.4, 1.45]

dt = 2000
steps = 20000

r0 = np.array([R, 0])
results = {}

for lam in lambdas:
    v0 = np.array([0, lam * v_circ])
    r, v = verlet(r0, v0, dt, steps)
    results[lam] = r

```

#### Plot the resulting trajectories


```python
plt.figure(figsize=(7,7))

for lam, r in results.items():
    plt.plot(r[:,0], r[:,1], label=f"λ={lam}")

plt.scatter([0],[0], color='yellow', s=250, label='Sun')
plt.title("Orbits for Different Initial Velocities")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.legend()
plt.show()
```


    
![png](output_35_0.png)
    


#### Compute orbital parameters for the eliptical cases


```python
print("Elliptical Orbit Parameters:\n")
for lam in [0.6, 0.8, 1.0, 1.2]:
    P, A, a, b, e = orbital_parameters(results[lam])
    print(f"λ = {lam}")
    print(f"  Perihelion P = {P:.3e} m")
    print(f"  Aphelion  A = {A:.3e} m")
    print(f"  Semi-major axis a = {a:.3e} m")
    print(f"  Semi-minor axis b = {b:.3e} m")
    print(f"  Eccentricity ε = {e:.4f}\n")
```

    Elliptical Orbit Parameters:
    
    λ = 0.6
      Perihelion P = 3.284e+10 m
      Aphelion  A = 1.496e+11 m
      Semi-major axis a = 9.122e+10 m
      Semi-minor axis b = 7.009e+10 m
      Eccentricity ε = 0.6400
    
    λ = 0.8
      Perihelion P = 7.040e+10 m
      Aphelion  A = 1.496e+11 m
      Semi-major axis a = 1.100e+11 m
      Semi-minor axis b = 1.026e+11 m
      Eccentricity ε = 0.3600
    
    λ = 1.0
      Perihelion P = 1.496e+11 m
      Aphelion  A = 1.496e+11 m
      Semi-major axis a = 1.496e+11 m
      Semi-minor axis b = 1.496e+11 m
      Eccentricity ε = 0.0000
    
    λ = 1.2
      Perihelion P = 1.496e+11 m
      Aphelion  A = 3.847e+11 m
      Semi-major axis a = 2.671e+11 m
      Semi-minor axis b = 2.399e+11 m
      Eccentricity ε = 0.4400
    


#### Construct an energy criterion and determine escape velocity


```python
def energy(v0):
    return 0.5 * v0**2 - G*M/R

vels = np.linspace(0.8*v_circ, 2*v_circ, 200)
energies = np.array([energy(v) for v in vels])

idx = np.where(energies > 0)[0][0]
v_escape_num = vels[idx]

print("Numerical escape velocity:", v_escape_num/1000, "km/s")

v_escape_theory = np.sqrt(2*G*M/R)
print("Theoretical escape velocity:", v_escape_theory/1000, "km/s")

```

    Numerical escape velocity: 42.153537932937844 km/s
    Theoretical escape velocity: 42.12786542722697 km/s


Both the simulated trajectories and the computed energies show that increasing the initial
velocity gradually changes the orbit from a closed ellipse to an escape trajectory. The
numerical escape velocity is found to be approximately 42 km/s, which agrees closely
with the theoretical value of about 42.1 km/s. This confirms that the velocity Verlet
method captures the correct energy behaviour for this system.

# Introduction to Q9

In Question 9 we were asked to simulate solar eclipses in a slightly simplified model of our solar system. A few assumptions allows us to make our modelling slightly simpler.

As well as this, we are given the criteria for an eclipse.

Initially we are given physical constants, such as the mass and radius of the earth, sun and moon as well as the gravitational constant G, initial conditions at 00:00:00, 21st of June 2010, Newton's Equations of motion and the velocity verlet algorithm.

# Modelling Assumptions and their Shortfalls.

Throughout this project we have to have some modelling assumptions that allow us to make our calculations, these come with consequences which are listed below:

### Assume we are in a heliocentric reference frame

- (The sun is in the center and stationary). In reality the gravity from other planetary objects in the solar system causes the sun to move and orbit the barycentre (the common centre of gravity of the solar system) so barycentric reference frame would have been more accurate.

### Assume the ommition of other planetary bodies. 

- This influences the orbit of the earth and the moon (and the sun but it's more of an issue of the reference frame!)

### (Implicitly) Assume Celestial bodies are perfectly spherical, uniform bodies. (Assumed by given radius and mass) 

- This throws off the shadow geometry that we use to predict eclipses later.

### (Implicitly) Assume Newtonian Gravity only included

- (Relativity is not mentioned).

### Assume the velocity verlet algorithm is completely accurate.

- It isn't, infact the error grows in proportion with the (time step)^{3}. So as the simulation progresses for long periods of time, estimates will become more inaccurate.

### Assume that the Umbra length and radius as well as Penumbra radius formulas are exact

- They are an approximation


# Code Overview:

We are given initial positions and velocities of the Earth and the Moon in a heliocentric reference frame. This implies that we can initialise the position vector of the Sun as the zero vector in three dimensions.

## Acceleration of the Earth and Moon

Directly from Newton's Equations, we can derive an equation for the acceleration of the Earth and the Moon and as acceleration is the second derivative of position with respect to time this becomes a differential equation for position.

Force of Earth:
\begin{align*}
m_e \frac{d^2 \vec{r_{e}}}{dt^2}
= G\, m_e m_s \frac{ \vec{r_s} - \vec{r_e} }{ |\vec{r_s} - \vec{r_e}|^3 } + G\, m_e m_m \frac{ \vec{r_m} - \vec{r_e} }{ |\vec{r_m} - \vec{r_e}|^3 }
\end{align*}
Force of the Moon:
\begin{align*}
m_m \frac{d^2 \vec{r_{m}}}{dt^2}
= G\, m_m m_s \frac{ \vec{r_s} - \vec{r_m} }{ |\vec{r_s} - \vec{r_m}|^3 } + G\, m_m m_e \frac{ \vec{r_e} - \vec{r_m} }{ |\vec{r_e} - \vec{r_m}|^3 }
\end{align*}

Dividing by $m_e$ and $m_m$ in each equation respectively and factorising gives us two second-order differential equations to solve in terms of position or an equation for acceleration.

Acceleration of the Earth:
\begin{align*}
\frac{d^2 \vec{r_{e}}}{dt^2}
= G\,(m_s \frac{ \vec{r_s} - \vec{r_e} }{ |\vec{r_s} - \vec{r_e}|^3 } + m_m \frac{ \vec{r_m} - \vec{r_e} }{ |\vec{r_m} - \vec{r_e}|^3 })
\end{align*}
Acceleration of the Moon:
\begin{align*}
\frac{d^2 \vec{r_{m}}}{dt^2}
= G\,(m_s \frac{ \vec{r_s} - \vec{r_m} }{ |\vec{r_s} - \vec{r_m}|^3 } + m_e \frac{ \vec{r_e} - \vec{r_m} }{ |\vec{r_e} - \vec{r_m}|^3 })
\end{align*}

This formula is what the functions accel_earth(rE, rM) and accel_moon(rE, rM) apply. (The arguments rE and rM being the position of the Earth and Moon respectively)

## The Velocity Verlet Algorithm

This algorithm is a numerical second order method of solving a differential equation. The equations are given by the booklet and programmed as the function verlet(rE, vE, rM, vM, dt):

\begin{align*}
r_{k+1} = r_k + v_kdt + \frac{1}{2}\frac{d^{2}\vec{r_k}}{dt^2}dt^{2}
\end{align*}
\begin{align*}
v_{k+1} = v_k + \frac{1}{2}(\frac{d^{2}\vec{r_k}}{dt^2} + \frac{d^{2}{\vec{r}}_{k+1}}{dt^2})dt
\end{align*}
notice that we use the second derivative of position which is just acceleration hence we can use our acceleration function within the velocity verlet function.

## Computing the Geometry of our model

Using acceleration and the Velocity Verlet algorithm we can compute the geometry of our model, and the resulting shadows.

Let S be the Sun and M be the Moon, since sunlight comes from the Sun we can say the vector $\vec{SM}$ is in the direction of sunlight (since sunlight comes from the Sun) and shadows are just the absence of light, they travel in the same direction. Converting this to a unit vector gives us the direction of the shadow axis:
\begin{align*}
\text{Direction of Shadow} = \frac{\vec{SM}}{||\vec{SM}||}
\end{align*}
Now we want to find out how much of the Earth lies in the direction of the shadow, using this we can find out if Earth is in the shadow cast by the Sun. as $\vec{a} \cdot \vec{b}$ = $\lVert \vec{a} \rVert \lVert \vec{b} \rVert \cos{\theta}$ when $\lVert \vec{b} \rVert = 1$ then this gives us the shadow's projection, how much of vector **a** lies in the direction of **b** (the shadow axis) by $\vec{a} \cdot \vec{b}$ = $\lVert \vec{a} \rVert \cos{\theta}$ Now because we have length of the projection, multiplying it by the shadow direction gives us the projection of the Earth onto the shadow axis. Intuitively, this projection is the side such that imagine the plane in which the Moon and the Earth exist, the hypotenuse of this triangle is the $\vec{ME}$ and the adjacent side is the projection, the angle is formed at the Moon. As this is the case, we can work out the perpendicular displacement from the Earth to the shadow axis by doing $\vec{ME}$ - projection and then by taking the magnitude gives us the perpendicular distance. Why is this important? Because it tells us how much of the Earth will be covered by shadow. Large perpendicular distances tell us that the shadow is unlikely to intersect Earth but a small perpendicular distance tells us that more of the Earth is covered by the shadow.

## Testing for an Eclipse

We can now test for an eclipse using this knowledge. Since we have the criteria for an eclipse, all we need to test is that there exists a portion of Earth that exists in either the umbra or penumbra. There exist approximate formulas for the Umbra Length, the Umbra Radius and the Penumbra Radius (We will assume they are exact as they are a very good approximation).
\begin{align*}
\text{Umbra Length} = \frac{d_{sm} r_m}{r_{s} - r_{m}}
\end{align*}
\begin{align*}
\text{Umbra Radius} = r_m\bigg(1-\frac{d_{me}}{L}\bigg)
\end{align*}
\begin{align*}
\text{Penumbra Radius} = r_m\bigg(1+\frac{d_{me}}{L}\bigg)
\end{align*} 
where:\
$d_{sm}$ = distance from the Sun to the Moon \
$d_{me}$ = distance from the Moon to the Earth\
$r_s$ = radius of the Sun\
$r_m$ = radius of the Moon \
(Veras D. 2019)

From the criteria of an eclipse, we know that a total eclipse is when the Earth intersects the umbral shadow. If the Earth intersects the penumbral shadow but not the umbral shadow then it is a partial or annular eclipse. A partial eclipse is when the umbral shadow exists but does not intersect Earth, but the Earth does lie in the penumbra.
\
If on the other hand the umbra can't reach the Earth but does lie in the penumbral shadow then it is an annular eclipse (the Moon blocks the Sun but not completely and is in the center so you see an annulus). If it doesn't intersect either then it's not an eclipse.
\
We can form the following logic and turn it into code by using if statements:

\begin{align*}
\text{Total Eclipse}: d_p < r_u + r_e \text{ and } r_u > 0
\end{align*}
\begin{align*}
\text{Annular Eclipse}: d_p < r_p + r_e \text{ and } r_u < 0
\end{align*}
\begin{align*}
\text{Partial Eclipse}: d_p < r_p + r_e \text{ and } r_u > 0
\end{align*}
where:\
$d_p$ = perpendicular distance of the Earth to the shadow axis.\
$r_u$ = Radius of the Umbra\
$r_p$ = Radius of the penumbra

## The Rough Scan

We technically have enough to run a computation as to where the eclipses may lie, but it will be incredibly computationally expensive. We choose to run 2 scans instead. A scan to check for eclipses using necessary but not sufficient conditions to determine if an eclipse can occur that is significantly less computationally expensive than a full scan.

We choose to use 2 conditions to whittle down candidates.

### 1. The Moon is between the Earth and the Sun

- The angle that the vector $\vec{MS}$ and $\vec{ME}$ needs to form is 180 degrees. It's less computationally expensive to compute the cosine of the angle as we can use the vector dot product to compute the cosine of the angle. and the cosine of 180 degrees corresponds to -1 so instead of having to use an arccos function we can just directly compare the cosine of the angle.
\
- To be completely exact we should check for $\cos{\theta}=-1$ however instead we check for $\cos{\theta}<-0.995$ this is to account for approximations made by our model. Initially this was programmed to be -0.9999 however the tolerance was reduced to be -0.995 as the code omitted partial eclipses in the scan.

### 2. The Moon is close to the Ecliptic Plane.

Initially we programmed the code without this condition, and the amount of possible candidates shot up to a couple thousand candidates instead of a few hundred. This condition helped us reduce the amount of false positives significantly.
\
Firstly, what is the ecliptic plane? It is the plane in which the Moon must intersect to be an eclipse, that is the plane in which the Earth is orbiting the Sun in. We can calculate the normal to this plane by considering the initial velocity and initial position vector of the Earth, we know these two vectors are coplanar, hence by taking the cross product we can get the normal of the plane. Dividing this by its magnitude gives us the unit normal vector of the plane.

\begin{align*}
h_E = \vec{r_e} \times \vec{v_e} \implies \hat{h}_E = \frac{h_E}{\lVert h_E \rVert}
\end{align*}

Using this we can define the ecliptic plane by the vectors that satisfy $r \cdot \hat{h}_E = 0$ since the vector $\hat{h}_E$ is a unit vector, taking the dot product of the position vector of the Moon and the unit normal of the ecliptic plane gives us the projection of the Moon onto the ecliptic plane, telling us how far the Moon is from the ecliptic plane. Ideally, we accept vectors such that $|r \cdot \hat{h}_E| = 0$ however to account for approximations we accept vectors such that $|r \cdot \hat{h}_E| < \text{tolerance}$ Intersection points to the ecliptic plane are called nodes, hence we've named in the code the tolerance as nodeTolerance. The value we chose was an acceptable balance between computational cost and accuracy.
\
If at time t the geometry adheres to both of these conditions we append the time to a list of times. We then integrate using the velocity verlet algorithm and check at the next time step. (Veras D. 2019)

## Cleaning the list

We then clean the list, removing times that correspond to the same eclipse. This will significantly reduce computation time by checking that the time is not the same as in the eclipses estimates list or within a day of the previous time.

## The Fine Scan

The fine scan is where we expend our computational resources to get an accurate estimate as to where our eclipses lie.\
Firstly from our cleaned list of times of eclipse estimates we check the times around it by a day, and then check the times around it in fine time steps.
But we have to check the position by integrating from our initial time and then getting to our region that we are checking, we integrate in coarse time steps outside the region we are concerned about (dt = 1800) we chose dt = 1800 in this region because any less and we significantly increase computation time and any higher and we decrease accuracy, dt = 1800 is a good compromise. Within the region we are concerned about our dt=10 which gives us a very accurate approximation of the geometry of our system, within the system we use the spatialGeometry function that we have defined to check every time step within the range that we are concerned about and check for the lowest perpendicular distance between the shadow axis and the Earth the lowest perpendicular distance will be our estimate for our central time. We append this to a list as well as the distance from the Moon to the Sun and the distance from the Moon to the Earth as well as the time at which this occurs. 

## Printing our eclipses

With our list, we input the elements of the list into our eclipse test function to find out what type of eclipse the fine scan list has given us.


# Additional Library Imports:

- Imported tqdm from the tqdm library to have a progress bar as the rough and fine scan's progress.
- Imported datetime and timedelta from the datetime library to convert seconds into dates.
- Imported time to compute the runtime of the program.

# Comparison of our Results to Real data (from NASA):
| Date       | Our Type | NASA Type | Our Time (approx) | NASA Central Time (TD) | Notes                                    |
| :--------- | :-------: | :-------: | :----------------: | :--------------------: | :-----------------------------------   |
| 2010-07-11 |   Total   |   Total   |      19:38:20      |        19:34:38        | **~+4 min**                            |
| 2011-11-25 |  Partial  |  Partial  |      06:49:30      |        06:21:24        | **~+28 min**                           |
| 2012-05-21 |  Annular  |  Annular  |      00:37:50      |    23:53:53 (May 20)   | **~+44 min (Across Midnight)**          |
| 2012-11-13 |   Total   |   Total   |      23:07:20      |        22:12:55        | **~+54 min**                           |
| 2013-05-10 |  Annular  |  Annular  |      01:32:50      |        00:26:20        | **~+66 min**                           |
| 2013-11-03 |  Annular  |   Hybrid  |      14:07:10      |        12:47:36        | **~+79 min, Hybrid Not classified**   | 
| 2015-03-20 |  Partial  |   Total   |      11:38:10      |        09:46:47        | **~+111 min, type mismatch**           |
| 2015-09-13 |  Annular  |  Partial  |      10:00:00      |        06:55:19        | **~+185 min, type mismatch**           |
| 2016-03-09 |   Total   |   Total   |      04:12:00      |        01:58:19        | **~+134 min**                          |
| 2016-09-01 |  Annular  |  Annular  |      11:42:40      |        09:08:02        | **~+155 min**                          |
| 2017-02-26 |  Annular  |  Annular  |      17:27:00      |        14:54:32        | **~+153 min**                          |
| 2017-08-21 |   Total   |   Total   |      21:31:50      |        18:26:40        | **~+185 min**                          |
| 2019-07-02 |   Total   |   Total   |      23:20:00      |        19:24:07        | **~+236 min**                          |
| 2019-12-26 |  Annular  |  Annular  |      08:54:20      |        05:18:53        | **~+215 min**                          |
| 2020-06-21 |  Annular  |  Annular  |      10:54:20      |        06:41:15        | **~+253 min**                          |
| 2020-12-14 |   Total   |   Total   |      20:23:00      |        16:14:39        | **~+248 min**                          |
| 2021-06-10 |  Annular  |  Annular  |      15:11:20      |        10:43:06        | **~+268 min**                          |
| 2021-12-04 |   Total   |   Total   |      12:12:30      |        07:34:38        | **~+278 min**                          |
| 2023-04-20 |  Annular  |   Hybrid  |      09:33:40      |        04:17:55        | **~+315 min, Hybrid Not Classified**  |
| 2023-10-14 |  Annular  |  Annular  |      23:17:50      |        18:00:40        | **~+317 min**                          |
| 2024-04-09 |   Total   |   Total   |      00:03:30      |        18:18:29        | **~+6 hrs (Across Midnight)**          |
| 2024-10-03 |  Annular  |  Annular  |      00:23:20      |        18:46:13        | **~+5.5 hrs**                          |
| 2025-03-29 |   Total   |  Partial  |      16:56:10      |        10:48:36        | **~+6 hrs, type mismatch**             |

(Espenak, 2014)

The error increase from start to finish can be attributed to the error of the velocity verlet algorithm being proportional to $(dt)^3$ as well as other modelling assumptions. The classification is correct for most eclipses aside from a select few cases. Below is a plot for the error growth:


```python
# Run this cell first
data = [
    (2010.53,  4/60),      # year, error hours
    (2011.90, 28/60),
    (2012.38, 44/60),
    (2012.87, 54/60),
    (2013.36, 66/60),
    (2013.84, 79/60),
    (2015.22, 111/60),
    (2015.70, 185/60),
    (2016.19, 134/60),
    (2016.67, 155/60),
    (2017.16, 153/60),
    (2017.64, 185/60),
    (2019.50, 236/60),
    (2019.98, 215/60),
    (2020.47, 253/60),
    (2020.95, 248/60),
    (2021.44, 268/60),
    (2021.92, 278/60),
    (2023.30, 315/60),
    (2023.78, 317/60),
    (2024.27, 360/60),
    (2024.76, 330/60),
    (2025.24, 360/60),
]

df = pd.DataFrame(data, columns=["year", "err_hours"])
```


```python
plt.figure(figsize=(8,5))
plt.plot(df["year"], df["err_hours"])
plt.xlabel("Year")
plt.ylabel("Error (hours)")
plt.title("Timing error of our model vs NASA")
plt.grid()
plt.show()
```


    
![png](output_48_0.png)
    


## Findings from Q9

In Question 9 we successfully simulated the orbits of the Earth and the Moon in a heliocentric reference frame under Newtonian gravity and the velocity verlet algorithm. Using our method we were able to correctly predict 23 solar eclipses to within a few hours of the eclipses actually occuring with only a few mismatches to the labels of the eclipses. The ability to solve the question demonstrates how we can relatively accurately predict solar eclipses using average computational power.\
\
Below is the code we used for Q9:


```python
G = 6.67348e-11

m_sun = 1.988420392e30
m_earth = 5.972000000e24
m_moon = 7.345828157e22

r_sun = 696340e3
r_earth = 6371e3
r_moon = 1737e3

# ======= Initial Conditions @ 21/6/2010 00:00:00 (All Given in a Heliocentric Reference Frame) =========== 

rE0 = np.array([-0.012083728, -1.394770664, -0.604680716]) * 1e11
rM0 = np.array([-0.015537064, -1.395982236, -0.605576290]) * 1e11

vE0 = np.array([ 2.930141099, -0.032094528, -0.013869403]) * 1e4 
vM0 = np.array([ 2.967343467, -0.121872473, -0.051163801]) * 1e4 

rS = np.zeros(3) # Sun Position fixed at the Origin

# ================ Maths pertaining to the Ecliptic Plane ========================

hE = np.cross(rE0, vE0) # The Normal Vector to the Ecliptic Plane
eclN = hE / lag.norm(hE) # The Unit Normal Vector to the Ecliptic Plane

# ======= Acceleration Functions (Provided Directly from Newton's Equations) ================

def accel_earth(rE,rM):
    rES = rS - rE # Vector going from Earth to Sun
    a = G * m_sun * rES / lag.norm(rES)**3

    rEM = rM - rE # Vector going from Earth to Moon
    a += G * m_moon * rEM / lag.norm(rEM)**3

    return a

def accel_moon(rE,rM):
    rMS = rS - rM # Vector going from Moon to Sun
    a = G * m_sun * rMS / lag.norm(rMS)**3

    rME = rE - rM #Vector going from Moon to Earth
    a += G * m_earth * rME / lag.norm(rME)**3

    return a

# ======== Velocity Verlet Integration ============

# Velocity Verlet Integration has an error proportional to (dt)^3 so as the time increases the estimates have a higher error margin

def verlet(rE,vE,rM,vM,dt): 
    
    # Current Accelerations
    
    aE = accel_earth(rE,rM)
    aM = accel_moon(rE,rM)
    
    # Update Positions
    
    rE_np1 = rE + vE * dt + 0.5 * aE * dt**2
    rM_np1 = rM + vM * dt + 0.5 * aM * dt**2

    # Update Accelerations

    aE_np1 = accel_earth(rE_np1, rM_np1)
    aM_np1 = accel_moon(rE_np1, rM_np1)

    # Update Velocities

    vE_np1 = vE + 0.5 * (aE + aE_np1) * dt
    vM_np1 = vM + 0.5 * (aM + aM_np1) * dt

    return rE_np1, vE_np1, rM_np1, vM_np1

# ========== Modelling for the Spatial Geometry of planetary motion =================

# The purpose of this function is to find whether the earth is inside the shadow cone of the moon.

def spatialGeometry(rE, rM): 

    # Vector Displacements
    
    rMS = rS - rM # Moon -> Sun Displacement vector
    rME = rE - rM # Moon -> Earth Displacement vector

    # Scalar Distances 
    
    sMS = lag.norm(rMS) # Sun -> Moon Distance Scalar
    sME = lag.norm(rME) # Moon -> Earth Distance Scalar

    shadowDirection = -rMS/sMS # Unit Vector in Direction of the Shadow Axis

    # Length of Projection

    projection_length = np.dot(rME, shadowDirection) # This tells us how far along the shadow axis does the earth lies, as shadowDirection is a unit vector
    # a.b = ||a||cos(theta) if b is a unit vector, which shows us that this is in this case rME being projected onto the unit vector of the shadowDirection
    
    projection = projection_length * shadowDirection # This tells us the component of the Displacement vector that is in the direction of the shadow

    pDisp = rME - projection # Perpendicular Displacement Vector
    pDist = lag.norm(pDisp) # Perpendicular Distance (This is the minimising distance between the Earth and the Shadow Axis)

    return sMS, sME, pDist

# ======== Testing for an eclipse ===========

def eclipseTest(sMS, sME, pDist, debug=False):

    # Umbra cone length
    L = sMS * (r_moon / (r_sun - r_moon))

    # Radii
    umbraR = r_moon * (1 - sME / L) # Umbra Radius
    penumbraR = r_moon * (1 + sME / L) # Penumbra Radius
    
# ------------ debug area ------------------------------
    if debug:
        print("ECLIPSE TEST:")
        print(" L =", L)
        print(" umbraR =", umbraR)
        print(" penumbraR =", penumbraR)
        print(" pDist =", pDist)
        print()
# ----------------------------------------------------
    
    # For the Moon to completely block the sun the Umbra's Radius MUST be greater than 0
    if umbraR > 0:
        if pDist < umbraR + r_earth: # We add the Radius of the Earth as the Earth is not a point but approximately a sphere
            return "Total Eclipse"
        if pDist < penumbraR + r_earth: # As the p_dist is greater than the umbra's Radius + the radius of the earth the moon is partially blocking the Sun
            return "Partial Eclipse"
        return None

    # Umbra does NOT reach Earth, so you can only have an Annular Eclipse or No Eclipse
    else:
        if pDist < penumbraR + r_earth: # Moon's shadow is smaller than the light cone of the sun, so creates an annulus
            return "Annular Eclipse"
        return None
        
# ============ Rough Scan ===================

# The purpose of the rough scan is to find out potential candidates for the eclipses, this prevents us from having to run the simulation finely for an insanely long time
# to whittle down computational resources, we then proceed by running a fine scan to deal with this.

def rough_scan(years=15, dt=300): #dt is in seconds

    print("Rough Scan has Started")
    
    seconds = years * 365.25 * 24 * 3600
    steps = int(seconds/dt)

    # Initialise Position and Velocity Vectors of the moon to their Initial vectors
    
    rE, vE = rE0.copy(), vE0.copy()
    rM, vM = rM0.copy(), vM0.copy()

    
    eclipseEstimates = [] # List Containing the Times of our eclipse estimates on our rough scan.

    for step in tqdm(range(steps),desc="Rough Scan",unit="Steps"): # Loops through the cases, appending in the list possible candidates for eclipses
        t = step * dt

        rMS = rS - rM # Vector from moon -> sun
        rME = rE - rM # Vector from moon -> earth

        cos_angle = np.dot(rMS, rME) / (lag.norm(rMS)*lag.norm(rME)) # This is the angle between the vector from the Moon to the Sun and the vector from the Moon to the Earth.
        
        colinear = (cos_angle<-0.995) # The colinear check, checks that the 2 vectors are approximately colinear and the vectors are in opposite directions to eachother
        # the reason the restriction is to <-0.995 is so that it takes into account the error from the velocity verlet integration.

        # Ecliptic Plane Geometry Checks - This Check is necessary to whittle down candidates as to be an eclipse the moon must intersect the ecliptic plane.
        # Without this check the number of possible candidates increases by over 10x So this significantly reduces the runtime and increases the accuracy
        sEcliptic = abs(np.dot(rM, eclN)) # Distance from Moon to the Ecliptic Plane
        nodeTolerance = 3e6 
        near_node = (sEcliptic < nodeTolerance) # A node is an intesecting point of the ecliptic plane, this checks that the moon is near the node to a reasonable degree
        if colinear and near_node: # All three checks must be successful in order for it to be a possible eclipse.
            eclipseEstimates.append(t)

        # Integrate to use for next step @ the next time step
        rE, vE, rM, vM = verlet(rE, vE, rM, vM, dt)

    print("Rough Scan Complete")
    
    return eclipseEstimates

# ========== Fine Scan ===============

# Since we already have the rough scan we can use the time values for our rough scan and look through the interval of the rough scan to work out the fine scan times.

def fine_scan(tc, window_hr=12, dt=10, dt_skip=300):

    # tc = time estimate, which is our center time.
    # window_hr = the window either side of tc in which it checks for the time of the solar eclipse in the fine scan
    # dt = The time steps within our window
    # dt_skip = The time steps outside our window
    
    t0 = max(0, tc - window_hr*3600) # Start Time (Max to prevent t0 being negative)
    t1 = tc + window_hr*3600 # End Time

    # Reset Initial Conditions
    rE, vE = rE0.copy(), vE0.copy()
    rM, vM = rM0.copy(), vM0.copy()

    t = 0 # Need to start @ t = 0 to do the integration as it's constantly moving.
    jump = 1800 # Jumps of 30 minutes is about optimal for as much accuracy while not sacrificing computational resources
    while t < t0 - jump:
        rE, vE, rM, vM = verlet(rE, vE, rM, vM, jump)
        t += jump

    # Then refine with dt_skip to integrate our way closer to t0 with minimal computational resources.
    while t < t0:
        step_dt = min(dt_skip, t0 - t)
        rE, vE, rM, vM = verlet(rE, vE, rM, vM, step_dt)
        t += step_dt

    
    t_ref = sMS_ref = sME_ref = None # Starts off with nothing but then gets appended as better estimation arises.
    pDist_ref = 1e20 # Refined Perpendicular Distance Estimate (which will get minimised) We start off super high and work our way down

    while t < t1:

        sMS, sME, pDist = spatialGeometry(rE,rM) # Computes the Geometry at the current t

        if pDist < pDist_ref: # Constantly keeps the most refined estimate for pDist and pDist_ref
            pDist_ref = pDist
            t_ref = t
            sMS_ref = sMS
            sME_ref = sME
                
        # Step Integrator
        rE, vE, rM, vM = verlet(rE, vE, rM, vM, dt)
        t += dt
        
    return t_ref, sMS_ref, sME_ref, pDist_ref

# ======== Convert seconds to datetime ========

initialTime = datetime(2010, 6, 21, 0, 0, 0) # Our Initial Time that we work from provided by the booklet.

# Following function converts the seconds into time
def dateConversion(t):
    return initialTime + timedelta(seconds=t) 

# ======== Computations ============

print("Run time has started")
start = time.time() # Starts timing the process
rough_times = rough_scan(years=15, dt=3600)
print("Number of Rough Candidates:",len(rough_times))

cleaned = [] # To avoid duplicate fine_scan calls to save computation time.
print("Cleaning Rough Scan List")
for t in rough_times:
    if not cleaned or abs(t-cleaned[-1]) > 86400: # The abs(t-cleaned[-1]) exists to only have one time to exist to be the representative for the eclipse
        cleaned.append(t)

rough_times = cleaned # Sets the rough times list as the cleaned list
print(f"Rough Scan list Cleaned to {len(rough_times)} Candidates")

refined_results = [] # List of the refined results

print("Fine Scan has Started")

for tc in tqdm(rough_times,desc="Fine Scan",unit="Candidates"):
    t_ref, sMS, sME, pDist = fine_scan(tc, window_hr=12, dt=10, dt_skip=300)
    refined_results.append((t_ref, sMS, sME, pDist)) 
print("Number of fine candidates:", len(refined_results))    
print("Fine Scan Complete")

# ======== Extract real eclipses from refined_results ========

def extract_eclipses(refined_results, debug=False):
    print("Extracting Eclipses")
    eclipse_list = [] # The list of eclipses

    for (t_ref, sMS, sME, pDist) in refined_results:
        etype = eclipseTest(sMS, sME, pDist) # Return the type of eclipse
        
# --------------------------- debug area ------------------------------------------
        if debug:
            print("t =",t_ref)
            print("sMS=",sMS)
            print("sME=",sME)
            print("pDist =",pDist)
            print("eclipseTest =", etype)
            print()
# -------------------------------------------------------------------------------------
        
        if etype is not None: # Eliminates Non-Eclipses to rid false positives
            eclipse_list.append((t_ref, etype))

    return eclipse_list

# =================== Print eclipse list =============================

def print_eclipses(eclipse_list):
    print(f"There are {len(eclipses)} predicted eclipses in this time period:")
    print("\nPredicted Eclipses (YYYY / MM / DD):\n")
    for t, etype in eclipse_list:
        print(f"{dateConversion(t)} | {etype:13}") # Prints the Date and Time and the type of eclipse


eclipses = extract_eclipses(refined_results)
print_eclipses(eclipses)
print("\nTotal Runtime:",time.time() - start, "seconds")
```

    Run time has started
    Rough Scan has Started


    Rough Scan: 100%|██████████| 131490/131490 [00:03<00:00, 40751.27Steps/s]


    Rough Scan Complete
    Number of Rough Candidates: 231
    Cleaning Rough Scan List
    Rough Scan list Cleaned to 23 Candidates
    Fine Scan has Started


    Fine Scan:  35%|███▍      | 8/23 [00:09<00:23,  1.56s/Candidates]

## References:

Espenak, F. (2014) Five Millennium Catalog of Solar Eclipses: -1999 to +3000 (2000 BCE to 3000 CE). NASA Eclipse Web Site. Available at: https://eclipse.gsfc.nasa.gov/SEcat5/SEcatalog.html
 (Accessed: 27 November 2025).

NASA Goddard Space Flight Center (2012) Eclipses and the Moon’s orbit. NASA Eclipse Web Site. Available at: https://eclipse.gsfc.nasa.gov/SEhelp/moonorbit.html
 (Accessed: 27 November 2025).

Veras, D. (2019) ‘Explicit relations and criteria for eclipses, transits, and occultations’, Monthly Notices of the Royal Astronomical Society, 483(3), pp. 3919–3952.
