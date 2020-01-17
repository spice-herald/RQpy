      Real Function C0(x,mu)
      Implicit none
      Real x,mu,xmin,xmax
      Real*8 A,B,C,D,Ratio,C00,x1,mu1,tmp
      Integer K,Niter
      Real mu0,p0
      Common/C0com/K,mu0,p0,Niter
      Integer M
      K=0
      If(mu.le.0.) Then
         C0=0.
         Write(6,*) "Illegal value of mu:",mu
         Return
      EndIf
      Call C0range(mu,xmin,xmax)
      If(x.lt.xmin) Then
         C0=0.
         Return
      Elseif(x.gt.xmax) Then
         C0=1.
         Return
      EndIf
      x1=x
      mu1=mu
      D=dexp(-x1)
      A=x1*D
      B=mu1*D
      M=mu1/x1 ! M is the largest integer < mu/x
      If(mu.gt.500.) Then
C If mu is large enough, the calculation can be done approximately
         C0=Dexp(-B)*( (1.D0+a*(1.D0-B) + .5D0*a*a*(B*B-5.D0*B+4.D0) )
     1      - D*(1.D0+a*(2.D0-B) + .5D0*a*a*(B*B-7.D0*B+9.D0) ) )
      Else
C Go ahead and do the calculation
         C00=1.D0 -D
         If(M.eq.1) Then
            C00=C00-D*(mu1-x1)
         ElseIf(M.gt.1) Then
             C=A/B
             Ratio=-B
             Do K=1,M-1
                tmp=Ratio*(  (1.D0-dfloat(K)*C)**K -
     1              D*(1.D0-dfloat(K+1)*C)**K  )
                C00=C00+tmp
               If(K.gt.10 .and. abs(tmp).lt.1.D-6 .and.
     1          B.lt..7*dfloat(K)) Go to 10
                Ratio=(-B)*Ratio/dfloat(K+1)
             EndDo
             C00=C00+Ratio*(1.D0-dfloat(M)*C)**M
 10          Continue
         EndIf
         C0=C00
      EndIf
      Return
      End
      Real Function dC0dx(x,mu)
      Implicit none
      Real x,mu,xmin,xmax
      Real*8 A,B,C,D,D2,tmp,C00,x1,mu1,Ratio
      Integer M
      Integer K,Niter
      Real mu0,p0
      Common/C0com/K,mu0,p0,Niter
       K=0
      If(mu.le.0.) Then
         Write(6,*) "Illegal value of mu:",mu
         dC0dx=0.
         Return
      EndIf
      Call C0range(mu,xmin,xmax)
      If(x.lt.xmin .or. x.gt.xmax) Then
         dC0dx=0.
         Return
      Endif
      x1=x
      mu1=mu
      D=dexp(-x1)
      D2=D*D
      A=x1*D
      B=mu1*D
      M=mu1/x1 ! M is the largest integer < mu/x
      If(mu.gt.1000.) Then
C If mu is large enough, the calculation can be done approximately
         C00=2.D0*D*(  (1.D0-B)*( 1.D0+A*(4.D0-B) )  )
         C00=C00-D2*( 2.D0-B+A*(B*B-7.D0*B+9.D0) )
         C00=Dexp(-B)*( C00-A*(B*B-3.D0*B+1.D0) +B )
      Else
C Go ahead and do the calculation
         If(M.lt.3) Then
            C00=D*(2.D0+mu1-x1)
            If(M.EQ.2) Then
               C00=C00-D2*(4.D0*(mu1-x1*2.D0) +2.D0 +(mu1-x1*2.D0)**2)
            EndIf
         Else
            C00=2.D0*D*(1.D0-D)
            Ratio=-B
            C=A/B
            Do K=1,M-2
               tmp=(2.D0*D*dfloat(K+1)*(1.D0-dfloat(K+1)*C)**K -
     1            D2*dfloat(K+2)*(1.D0-dfloat(K+2)*C)**K -
     2            dfloat(K)*(1.D0-C*dfloat(K))**K)*Ratio
               C00=C00+tmp
               If(K.gt.10 .and. abs(tmp).lt.1.D-6 .and.
     1          B.lt..7*dfloat(K)) Go to 10
               Ratio=(-B)*Ratio/dfloat(K+1)
            EndDo
            tmp=(dfloat(2*M)*D +
     1        B*(1.D0-dfloat(M)*C))*(1.D0-dfloat(M)*C)**(M-1)
            tmp=tmp-dfloat(M-1)*(1.D0-dfloat(M-1)*C)**(M-1)
            C00=C00+Ratio*tmp
 10         Continue
C           Write(6,*) K
         EndIf
      EndIf
      dC0dx=C00
      Return
      End
      Subroutine C0range(mu,xmin,xmax)
C For a given mu finds the x range over which C0 and dC0/dx differ
C significantly from zero.
      Implicit None
      Real mu,xmin,xmax,logmu
      If(mu.le.10.) Then
         xmin=.12*(mu**.75)
         xmax=mu
      Else
         logmu=log(mu)
         xmin=logmu -3. + 5./(logmu*logmu) + 3.5/mu
         If(mu.lt.17.) Then
            xmax=mu
         Else
            xmax=logmu + 15.7 -30./mu
         EndIf
      Endif
      Return
      End
      Real Function C0Int(A,B,mu1,mu2,z,Icode)
C Uses Simpson's rule to integrate dC0(x,mu1)/dx * C0(z-x,mu2) from
C A to B.
C Icode=0 means "normal" integration
C Icode=1 means part of the range had C0=1
C Icode=2 means all of the range with dC0/dx > 0 had C0=1
C Icode=3 means zero integrand everywhere
C Icode=4 means A .ge. B or one of the mu's or z is negative
C        
      Implicit none
      Real A,B,mu1,mu2,z,C0,dC0dx
      Integer Icode
      Real x,xmin,xmax,xmin1,xmax1,xmin2,xmax2,dx,twodx,Two,Four
      Integer I,N/1000/ ! 2N+1 values of x in the Simpson's sum.
      If(A.ge.B .or. mu1.le. 0. .or. mu2.le. 0. .or. z.le. 0.) Then
         Write(6,*) "Bad input to C0Int"
         C0Int=0.
         Icode=4
         Return
      EndIf
C Find the range over which the integral is to be done.
      Call C0range(mu1,xmin1,xmax1)
      Call C0range(mu2,xmin2,xmax2)
      xmin=max(A,xmin1)
C If z-x > xmax2 C0(z-x,mu2)=1.  I.e. C0=1 if x < z-xmax2
      xmax=min(B,xmax1,z-xmin2)
C It's ok if z-x is so large (maybe even larger than mu2) that
C C0(z-x,mu2) = 1.  x should be included in the integral so long as neither
C dC0/dx nor C0 is essentially zero.  But if C0=1, it's easy to integrate
C dC0/dx * C0, so that part (small x) shouldn't be numerically integrated.
      C0Int=0.
      Icode=3
      If(xmin.lt.xmax) Then
         Icode=0
C For x < z-xmax2 we have z-x > xmax2, so C0(z-x,mu2)=1.
         If(z-xmax2 .gt.xmin) Then
            If(z-xmax2.lt.xmax) Then
               Icode=1
               C0Int=C0(z-xmax2,mu1)-C0(xmin,mu1)
               xmin=z-xmax2
            Else
               Icode=2
               C0Int=C0(xmax,mu1)-C0(xmin,mu1)
               Return
            EndIf
         EndIf
C dx = distance between adjacent values of x.
C xmin                           xmax
C  |                             |
C  1  4  1     1  4  1     1  4  1   times function values  +
C        1  4  1     1  4  1         times function values  =
C  1  4  2  4  2  4  2  4  2  4  1   times function values (2N+1 values of x)
C  |twodx|twodx|                     (xmax-xmin) = N*twodx
C     1     2      ...        N
C Simpson's rule adds together approximate average function values over each of
C the N twodx long intervals.  This pattern makes 1+4+1 times function values
C in each twodx interval, while for an integral we want 2*dx*(mean value of
C the function in the 2*dx interval).  So multiply the sum by twodx/6.
C
C    Note that C0 and dC0/dx are not analytic.  For example, dC0(x,mu)/dx has a
C discontinuity at x=mu/2.  So Simpson's rule might be no better than the
C pattern 1 1 1 1 1 1 1 +
C           1 1 1 1 1   =
C         1 2 2 2 2 2 1
C with the sum multiplied by dx/2 to make it into an integral.
         twodx=(xmax-xmin)/float(N)
         dx=.5*twodx
         x=xmax-dx
         Four=dC0dx(x,mu1)*C0(z-x,mu2)
         Two=0.
         do I=1,N-1
            x=xmin+twodx*float(I)
            Two=Two+dC0dx(x,mu1)*C0(z-x,mu2)
            x=x-dx
            Four=Four+dC0dx(x,mu1)*C0(z-x,mu2)
         EndDo
         C0Int=(4.*Four+2.*Two+dC0dx(xmin,mu1)*C0(z-xmin,mu2)
     1        + dC0dx(xmax,mu1)*C0(z-xmax,mu2))*(dx/3.) + C0Int
      EndIf
      Return
      End
      Real Function XC0(p,mu,ICode)
C Finds the value of x for which C0(x,mu)=p
C Needs the real function RZERO(A,B,x_0,R,EPS,MAXF,F), where F(x,I) =
C C0(x,mu)-p, and I is supplied by RZERO: 1=first reference, 3=final reference
C ICode=0: all is well
C ICode=1: p out of range
      Implicit None
      Common/C0Com/K,mu0,p0,Niter
      Integer K,Niter
      Real p,mu,p0,mu0
      External FXC0
      Real EPS/0.00001/,R,cut
      Integer MAXF/50/,Icode
      Icode=0
      cut=1.-exp(-mu)
      If(p .le. 0. .or. p .ge. cut-EPS) Then
         Niter=0
         If(p.le.0.) then
            XC0=0.
         Else
            XC0=mu
         EndIf
         If(p.gt.1. .or. p.lt.-EPS) ICode=1
         Return
      EndIf
      mu0=mu
      p0=p
      Call RZERO(EPS,mu,XC0,R,EPS,MAXF,FXC0)
      Return
      End
C
      Real Function FXC0(x,I)
      Implicit None
      Common/C0Com/K,mu,p,Niter
      Real mu,p,x,C0
      Integer I,K,Niter
      If(I.eq.1) Then
         Niter=0
      Else
         Niter=Niter+1
      EndIf
      FXC0=C0(x,mu)-p
      Return
      End
