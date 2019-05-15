from rlmodel import *
from  sim import *
def main():

    wind = array([2, -1.57])
    
    # initial state
    x = array([[10,-40,-3,1,0]]).T   #x=(x,y,θ,v,w)

    train(num_episodes = 50)
    ax=init_figure(-100,100,-60,60)

    target = array([-15,20])
    for t in arange(0,200,0.2):
        clear(ax)
        u = select_action()
        print(u)
        x, δs = step(x, u, 0.1, wind)
        plot(target[0], target[1], '*b')
        draw_sailboat(x,δs,u[0,0],wind[1],wind[0])
        update(x[0:2,0], target, wind)


if __name__ == '__main__':
    main()