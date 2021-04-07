import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 

# Initial condition. Randomly distributed speakers.

def initial_cond(m: int, n: int, pa: float = 0.5) -> np.array:
    """ Creates the initial (mxn) array. 
    
    Each point within the (mxn) array represents a citizen speaking 
    language A or language B. The default initial condition corresponds 
    to a scenario where the speakers are randomly distributed, the 
    probability of each speaker being p(A) = pa and p(B) = 1 - pa. 
    
    Args:
        m: Size of the array.
        n: Size of the array.
        pa: Probability that a single node within the array speaks 
            language A. Defaults to 0.5.
    
    Returns:
        Returns a np.array(shape=(m, n)) where each node speaks either 
        language A (represented by a value 1) or language B (represented
        by a value -1)

    """
    popu = np.ones(shape=(m, n))
    mask = np.random.rand(m,n)
    popu[(mask <= (1.0 - pa))] = -1     # Speakers language B
    return popu 


def repres(popu: np.array) -> plt.figure:
    """Array representation. 
    
    Graphical 2D-representation of the (mxn) array. Each site represents
    an individual speaking either language A or language B. Languages 
    are pictured by a binary selection of colors (blue, red). 
    
    Args:
        popu: Array containing the language spoken by the population. 
            The values contained inside the array are 1 (lang A) and 
            -1 (lang B). 
            
    Returns: 
        Returns a mpl.figure object containing the graphical representation
            of the array. 
    
    """
    col = mpl.colors.ListedColormap(['Blue','Red'])
    colbar_tick = np.array([ -1,1 ])
    fig = plt.figure()
    ax  = plt.axes()
    plot = ax.matshow( popu , cmap = col , origin = 'lower' , animated=True)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_major_formatter(lambda val, pos : r'{}'.format(int(val)+1) )
    ax.yaxis.set_major_formatter(lambda val, pos : r'{}'.format(int(val)+1) )
    fig.colorbar(
        plot,ax=ax, 
        ticks=colbar_tick , 
        label='Language'
        ).ax.set_yticklabels(['B', 'A'])
    # To avoid an excessive computation cost, the graphical
    # representation of the lattice is not displayed. Only the figure instance 
    # is returned. 
    plt.close()
    return fig 


def periodic_boundary(index: tuple, lattice_shape: tuple) -> tuple:
    """Periodic boundary conditions. 
    
    We consider a regular lattice with periodic boundary conditions.
    per_bon function is used to apply this condition to the selected 
    node. If the node is located outside the boundaries, the neighbors 
    are selected applying usual nearest neighbor conditions. If the 
    node is located at the boundary, the function will return a tuple 
    where the neighbors are selected following periodic boundary 
    conditions (in this way, the regular lattice becomes a torus). 

    Args:
        index: 2D-tuple. Represents the selected node. 
        lattice_shape: 2D-tupple. Shape of the lattice. 
    Returns:
        The functions returns a tuple where periodic boundary conditions
        are applied if needed.
    
    Examples:
    If the lattice is a 5x5 array, a node located at position (6, 5) will
    result:
    
    >>> print(per_bon((6, 5), (5, 5)))
    (1,0)
    """
    return tuple((i%s for i, s in zip(index, lattice_shape)))


def language_dynamics(popu: np.array,
                      m: int, 
                      n: int, 
                      s: float,
                      a: float = 1.0) -> np.array:
    """ Language change. 
    
    Language dynamics. Evolution of the number of speakers of each
    language. Each time this function is called, it computes the
    probability of change for a selected node. 
    The probability to change from language A to B is: 
    pAB = (1 - s) * nB**a 
    The probability to change from language B to A is: 
    pBA = s * nA**a 
    The steps are the following:
    1) A random node is selected.
    2) The language of each neighbor surrounding the node is computed, 
    counting the number of speakers. 
    3) The probability of change is computed. If the probability is 
    bigger than a uniformly distributed random number, the language 
    of the selected node is changed.
    
    Args: 
        popu: Array containing the language spoken by the population. 
            The values contained inside the array are 1 (lang A) and 
            -1 (lang B). 
        m: Size of the array.
        n: Size of the array.
        s: Status or prestige of the language A.
        a: Volatility of the system. Determines the location of the 
            fixed points. Defaults to 1.0 
    
    Returns: 
        The function returns the updated version of the np.array popu. 
    """
    # Selection random node. 
    ii = np.random.randint(m)
    jj = np.random.randint(n)
    lang = popu[ii,jj] 
    # Neighbors language
    nn = np.array([]) # Array containing the neighbors 
    nn = np.append(nn, popu[periodic_boundary((ii-1, jj), (m, n))])  # upper 
    nn = np.append(nn, popu[periodic_boundary((ii+1, jj), (m, n))])  # lower 
    nn = np.append(nn, popu[periodic_boundary((ii, jj-1), (m, n))])  # left 
    nn = np.append(nn, popu[periodic_boundary((ii, jj+1), (m, n))])  # right 
    nn = nn.astype(np.int)
    nA = np.count_nonzero( nn > 0 )/4.0 # Number A speakers
    nB = 1.0 - nA                       # Number B speakers 
    # Language dynamics 
    v = np.random.uniform()
    if lang == 1: 
        pab = (1.0-s)*nB ** a
        popu[ii,jj] = -1  if (v < pab) else popu[ii,jj]
    else:
        pba = s*nA ** a
        popu[ii,jj] = 1 if (v < pba) else popu[ii,jj]
    
    return popu
   
   
# Parameters 
m = 20
n = 20
a = 1.0
s = 0.75
pa = 0.35
# Initial condition 
popu = initial_cond(m, n, pa)
initial_popu = popu 
# Check error
na = np.count_nonzero(popu[popu > 0])
nb = np.count_nonzero(popu[popu < 0])
print('error') if nb != (m*n - na)  else None
# Iterative process 
mult = 15 
maxit = mult * 300      # Maximum number of iterations. Information is stored
                        # every mult steps 
numsp = np.ones(shape=(int(maxit / mult), 2))       # Container   

for iterr in np.arange(maxit):
    popu = language_dynamics(popu, m, n, s)
    # Saving process and number of speakers 
    numa = np.count_nonzero(popu[popu > 0])
    numb = np.count_nonzero(popu[popu < 0])
    if (iterr%mult) == 0:
        ph = int(iterr / mult)
        numsp[ph] = np.array([numa, numb])
        #figr = repres(popu)
        #figr.savefig('lang_comp_{}.png'.format(str(ph)),bbox_inches='tight')
    else:
        None
    # Exit process. We can stop if one of the languages dies
    if (numa == 0 or numb == 0):
        break
    else:
        None

# Representation 
#fig , (ax1, ax2, ax3) = plt.subplots(1, 3)
#fig.suptitle('Language competition')
if __name__ == "__main__":
    #figr = repres(popu)    
    #dummy = plt.figure()
    #new_manager = dummy.canvas.manager
    #new_manager.canvas.figure = figr
    #figr.set_canvas(new_manager.canvas)
    #plt.show(figr)
# Representation of the lattice final state 
    #figr=repres(popu)
    #dummy = plt.figure()
    #new_manager = dummy.canvas.manager
    #new_manager.canvas.figure = figr
    #figr.set_canvas(new_manager.canvas)
    #plt.show()

    fig, ax = plt.subplots()
    ax.plot( numsp[:,0] , c='red' , linestyle='solid', marker='o' , 
    linewidth=1.5 , markersize=5.5 , label='Lang A')
    ax.plot( numsp[:,1] , c='blue', linestyle='solid', marker='x' , 
    linewidth=1.5 , markersize=5.5 , label='Lang B')
    ax.xaxis.set_major_formatter(lambda val, pos : r'{}'.format(int(val)*mult))
    ax.set(xlabel = 'Iteration', 
           ylabel = 'Number speakers', 
           title = 'Number of speakers as a function of time'
           )
    ax.legend(loc='upper left')
    plt.show()
