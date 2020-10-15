function plotline(w,mindatax1,maxdatax1)
    x = linspace(mindatax1,maxdatax1,100);
    y = -(w(2)/w(3))*x - w(1)/w(3);
    plot(x,y);
    return
    