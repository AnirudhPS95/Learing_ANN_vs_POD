program LidDriven

implicit none

!Variable declaration===================================================
integer ::i,j,k,q,m,n,endtime,cnt
double precision ::L,H,ptol,tol,dx,dy,dt,A,B,Rey,s,et
double precision, dimension(:), allocatable ::x,y
double precision, dimension(:,:), allocatable ::U,V
double precision,  dimension(:,:,:), allocatable ::P,W


!User input=============================================================
!print*, "Enter the dimension of Length and Height respectively"
!read*, L,H
L=1; H=1;
print*, "Enter the number of nodal points along the Length and Height"
read*, m,n
!print*, "Enter the tolerance level for pseudo-iteration"
!read*, ptol
!print*, "Enter the tolerance level for real solution"
!read*, tol
print*, "Enter the time for which you want to run the program"
ptol=1e-5; tol=1e-5
read*, et

!Allocating array=======================================================
allocate(x(m)); allocate(y(n));
allocate(U(m,n)); allocate(V(m,n));
allocate(P(m,n,2)); allocate(W(m,n,2));

!open(10, file='hello.txt', status='unknown')

print*,"hello"

!Mesh Generation========================================================
dx=L/(m-1); dy=H/(n-1);
x(1)=0; do i=1,m; x(i+1)=x(i)+dx; end do;x(m)=1;
y(1)=0; do i=1,n; y(i+1)=y(i)+dy; end do;y(n)=1;
open(11,file='my.txt', status='unknown')
do i=1,m
    do j=1,n
        write(11,*),x(i),y(j)
    end do
end do
close(11)
pause

!Initialisation=========================================================
U(:,:)=0.0; V(:,:)=0.0; P(:,:,:)=0.0; W(:,:,:)=0.0;

!Boundary conditions====================================================
W(:,n,:)=-2/dy   !All other are zero as stream function is assumed zero


!CFL criterion used for time calculation================================
dt=0.0001  !Extra 0.5 is for Factor of safety=====================
endtime=et/dt
print*,endtime


!Useful parameters======================================================
A=dt/(2*dx); B= dt/(dx)**2; !For this problem it stays =1
Rey=400;





!Solution loop starts===================================================
cnt=0
!Time loop---------------------------------------------------------------
do q=1,endtime
    W(:,:,1)=W(:,:,2)
    !Momentum conservation----------------------------------------------
    do j=2,n-1
        do i=2,m-1
            W(i,j,2)=W(i,j,1)-A*( U(i,j)*(W(i+1,j,1)-W(i-1,j,1))+V(i,j)*(W(i,j+1,1)-W(i,j-1,1)))+&
                              (B/Rey)*(W(i+1,j,1)+W(i-1,j,1)+W(i,j+1,1)+W(i,j-1,1)-4*W(i,j,1))
        end do
    end do

    !Mass conservation with pseudo iterations as equation is Parabolic---
    do k=1,1000000
        do j=2,n-1
            do i=2,m-1
                P(i,j,1)=P(i,j,2)
                P(i,j,2)=(dx**2)*0.25*W(i,j,2)+0.25*(P(i+1,j,2)+P(i-1,j,2)+P(i,j+1,2)+P(i,j-1,2))
            end do
        end do
        s=0
        do j=2,n-1
            do i=2,m-1
                s=s+(P(i,j,2)-P(i,j,1))**2
            end do
        end do

        if (sqrt(s/((m-1)*(n-1)))<ptol) then
            EXIT
        end if

    end do


    !Velocity calculation using stream newly calculated stream function------
    do j=2,n-1
        do i=2,m-1
            U(i,j)=(P(i,j+1,2)-P(i,j-1,2))/(2*dy)
            V(i,j)=-(P(i+1,j,2)-P(i-1,j,2))/(2*dx)
        end do
    end do

    !Boundary condition application---------------------------------
    do j=2,n-1
        W(1,j,2)=-2*P(2,j,2)/(dx**2)
        W(m,j,2)=-2*P(m-1,j,2)/(dx**2)
    end do

    do i=2,m-1
        W(i,1,2)=-2*P(i,2,2)/(dy**2)
        W(i,n,2)=-(2*P(i,n-1,2)+2*dy)/(dy**2)
    end do

    s=0
    do j=2,n-1
        do i=2,m-1
           s=s+(W(i,j,2)-W(i,j,1))**2
        end do
    end do
    if (sqrt(s/(n-1)*(m-1))<tol) then
        EXIT
    end if
    cnt=cnt+1
    if (mod(cnt,10000)==0)then
        print*, "Iteration",cnt,"=========================================="
        print*,dt*cnt, P((m+1)/2,(n+1)/2,2),U((m+1)/2,(n+1)/2)
    end if




end do
print*,cnt

print*,"hello"


do i=1,n
    do j=1,m
        write(10,*), x(i)/H, y(j)/L, P(i,j,2)
    end do
end do

close(10)

print*,"hello"
deallocate(x)
deallocate(y)
deallocate(U)
deallocate(V)
deallocate(W)
deallocate(P)

end program
