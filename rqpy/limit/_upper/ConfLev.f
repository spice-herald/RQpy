      Real Function ConfLev(m,x,mu,icode)
C ConfLev(m,x,mu,icode) = C_m(x,mu):
C Evaluate the probability that the maximum expected contents of an interval
C with at most m points is less than x given that the total range is mu.
C The return code, icode, is 0 if all is ok, 2 if non-allowed parameters,
C icode=1 if there was a possibly poor extrapolation.
C
C Uses GAMDIS, which may require cern libraries libmathlib and libkernlib
C Needs table Cm.txt.  At present the table covers CL = 0.0001 to .9999
C and mu << .01 to mu=104.585 and m=0 to 100.  m=100 is enough for CL > 0.0001
C for mu < 66.68635, it's enough for CL > 0.001 for mu < 73.69981, it's
C enough for CL > 0.12 for mu < 90.01713, and it's enough for CL > 0.65 for
C all included mu.
C
C When quadratic interpolation is done, I saw for C0 an error > .002, but
C that's rare (try with m=0, x=.501, mu=1).
      Implicit None
      Integer m,icode
      Real x,mu,C0
C
C Tabulated information
      Integer NMax,N,Nmeans,NCLs,NCLs2,Ntrials
      Parameter(NMax=100)
      Parameter(Nmeans=186)
      Parameter(NCLs=84)
C     Parameter(NCMaxmu=70)
C      Common/CLtab/CL,Mean,xcl,Nxcl,muval,CMaxval,meanlog
C Values to be saved between calls can go in CLtab:
      Common/CLtab2/CL,Mean,xcl,Nxcl,meanlog,CLmumu
      Real CL(NCLs),xcl(0:NMax,NCLs,Nmeans),mean(Nmeans),
     1 meanlog(Nmeans),CLmumu(Nmeans,0:NMax)
C     Real muval(NCMaxmu),CMaxval(NCMaxmu)
      Integer Nxcl(0:NMax,Nmeans)
C
      Real mulog,GAMDIS,p(0:2)
      Logical first/.true./,NewGroup/.true./
      Integer I,I0,II,J,K,LL,KMid,KLow,KHigh
      Real x0,x1,x2,p0,p1,p2,b,z,d0,d1,d2,mu0(0:2)
      Logical Debug/.false./
      If(first) Then ! Input the table
C xcl(m,J,I) is the maximum interval containing m points
C corresponding to mu=mean(I) and confidence level CL(J).  If Nxcl(m,I)=0,
C then the m is so large compared with mu that one hardly ever gets m
C events; so if x<mu, the confidence level is very small.  I assume that if
C xcl(m,J,I) is meaningful then so is xcl(m,J,I+1) and xcl(m,J-1,I)
C (unless J=1).
        First=.false.
        Do I=0,NMax
           Do J=1,Nmeans
              Nxcl(I,J)=0
           EndDo
        EndDo
        LL=0
        Open(21,file='Cm.txt',status='OLD',form='FORMATTED')
        Do I=1,Nmeans
 5        If(NewGroup) Then ! Allows re-input when debugging.
            NewGroup=.false.
            Read(21,10) Ntrials,N,NCLs2
 10         Format(I9,2I3)
            If(NCLs2.ne.NCLs .or. N.le.0 .or. N.gt.NMax) Then
              Write(6,*) "Bad Cm.txt, with NCLs2, N: ",NCls2,N
              Stop
            EndIf
            Read(21,14) CL
 14         Format(10F8.5)
          EndIf
          Read(21,16) mean(I),(Nxcl(K,I),K=0,N)
 16       Format(F12.5,(30I3))
          If(mean(I).gt.0.) Then
            Do K=0,N
             II=Nxcl(K,I)
             CLmumu(I,K)=GAMDIS(mean(I),float(K+1))
             If(II.gt.0) Then
                Read(21,17) (xcl(K,J,I),J=1,II)
                If(CL(II).gt.CLmumu(I,K)) Then
C This can occasionally happen, I guess because of roundoff error.
                   If(Debug) Then
                      Write(6,18) K,xcl(K,II,I),mean(I),CL(II),
     1                 CLmumu(I,K)
 18                   Format('Cm(x,mu) for m =',I2,', x =',F9.5,
     1              ' and mu =',F9.5,' is',F9.6,' >',F9.6,
     2              ', the maximum.')
                   EndIf
                   Nxcl(K,I)=II-1
                   LL=LL+1
                EndIf
             EndIf
 17          Format(10F9.5)
            EndDo
            meanlog(I)=log(mean(I))
          Else
            NewGroup=.true.
            Go to 5
          EndIf
        EndDo
        Close(21)
        If(LL.gt.0 .and. Debug) Then
           Write(6,20) LL
 20        Format(I5,' table corrections needed for excess CL.',/)
        EndIf
      EndIf ! Completion of table input.
C Make some simple checks on reasonableness of the input parameters.  If
C they are unreasonable, either return a reasonable output anyway, or give up.
C Allow mu from 0 to mean(Nmeans)=end of the table.  So extrapolate to
C mu < mean(1) but not to mu > mean(Nmeans).
      Icode=0
      If(mu.le.0. .or. m.lt.0 .or. x.le.0.) Then
         ConfLev=0. ! ok if mu.ge.0 and m.ge.0
         If(mu.lt.0. .or. m.lt.0) Go to 1000
      ElseIf(mu.gt.mean(Nmeans) .or. m.gt.NMax) Then
C The table hasn't the needed information, so give up.
         Go to 1000
      ElseIf(x.gt.1.00001*mu) Then
C It's reasonable for x to be mu if the interval has the whole range,
C but x shouldn't ever be definitely > mu.  If it is, that's probably
C a mistake, but since mathematically Cm(x,mu) for x>mu is defined to be 1,
C return 1.
         ConfLev=1.
      ElseIf(x.gt. 0.99999*mu) Then
         ConfLev=GAMDIS(mu,float(m+1))
      ElseIf(m.eq.0 .and. .not. Debug) Then
C C0 is best if m=0, and there are no tables for mu<mean(1).
C But for debugging, treat m=0 the same as m>0.
         ConfLev=C0(x,mu)
C From now on, m .ge. 1 and x is in the range 0+ to mu-.
      ElseIf(mu.lt.mean(1)) Then
C For the current tables, mu<.01 and m>0 has ConfLev < .0001, so it's
C ok to just take it that or, slightly better, interpolate
C between x=0 and x=mu-.  If m>4 and mu<mean(1)=.01, ConfLev is extremely
C small.  And if we know ConfLev < 1.E-6, that's considered extremely small.
         If(m.gt.4 .or. mu.lt.1.e-6) Then
            ConfLev=0.
         Else
            ConfLev=((x/mu)**m)*GAMDIS(mu,float(m+1))
         EndIf
      Else ! The table might be valid for this m,mu
C Find which mean(I) is just below mu.  Since mu is less than mean(Nmeans),
C the I must be less than Nmeans; so there is information for I+1.
         mulog=log(mu) ! Since mu.ge.e^{-92*.05}, 20*mulog is at least -92.
         I0=20.*mulog + 93.  ! I0 is 1 just above the lowest mu.
         If(I0.lt.1) Then
            I0=1  ! Don't let I0 get below 1.  This should never happen.
            Icode=1
         EndIf
         If(I0+2.gt.Nmeans) I0=Nmeans-2  ! Don't let I0+2 get > Nmeans
C For I0, I0+1, and I0+2 find CL at the given value of x; then quadratically
C interpolate unless we expect Nxcl=0.
C If Nxcl = 0 then linearly interpolate between x=mu with CL=GAMDIS(mu,m+1)
C and x=0 with CL=0.  Otherwise quadratically or exponentially interpolate.
C Method of exponential interpolation:
C If x<xcl(m,1) for Nxcl>0, take x0=xcl(m,1) and take x1=mu
C if Nxcl=1 and x1=xcl(m,2,I) for Nxcl>1:  Have x0, x1, p0=p(x0), p1=p(x1).
C Take x = x0/(1+by(p)) where y(p) = log(p0/p) is a choice motivated by
C noticing that x near zero changes by roughly constant amounts when
C C changes by a factor of 2.  The constant "b" is (x0/x1 - 1)/y(p1).
C It's easily verified that this form is correct at x -> 0, x0, and x1.
C Then p = p0*exp(z/x), where z = (x-x0)/b.  To avoid computer overflow
C or underflow, if x < -.05*z then don't evaluate the exponential, but instead
C take p = .0025*exp(-20) = 0.515E-11.  So far, I've found this method
C works better than linear approximation for ConfLev < 0.2.
         If(Nxcl(m,I0) .eq. 0 .and. Nxcl(m,I0+1) .eq. 0) Then
C Use a crude extrapolation.  ConfLev should be < 0.0001.
            ConfLev=((x/mu)**m)*GAMDIS(mu,float(m+1))
C At first I had (x/mu) instead of (x/mu)**m, but the latter works better.
         Else
C Estimate for I0, I0+1, and I0+2; then quadratically interpolate.
            Do J=0,2
               I=I0+J
               mu0(J)=mean(I)
               If(x.ge.mean(I)) Then
                  mu0(J)=x
                  p(J)=GAMDIS(x,float(m+1))
               ElseIf (Nxcl(m,I) .eq. 0) Then
                  p(J)=((x/mean(I))**m)*CLmumu(I,m)
C From now on, Nxcl .ge. 1
               ElseIf(x.le.xcl(m,1,I)) Then
C x is below the lowest tabulated one (so CL < .0001 for the current table).
C make p(J) exact for x=0, x0, and x1, where x0=lowest MC x for the m,I
C and x1 is either the second lowest, if it exists, or mu.  Use exponential
C extrapolation.
                  x0=xcl(m,1,I)
                  p0=CL(1)
                  If(Nxcl(m,I).eq.1) Then
                     x1=mean(I)
                     p1=CLmumu(I,m)
                  Else
                     x1=xcl(m,2,I)
                     p1=CL(2)
                  EndIf
                  b=(x0-x1)/(x1*log(p0/p1))
                  z=(x-x0)/b
                  If( x .lt. -(.05*z) ) then ! avoid underflow
                     p(J)=0.515E-11
                  Else
C p(J) is exactly 0, p0, and p1 for x=0, x0, and x1.
                     p(J)=p0*exp(z/x)
                  EndIf
C From now on, Nxcl .ge. 1 and x .ge. xcl(m,1,I), the lowest MC tabulated x.
               ElseIf(x.ge.xcl(m,Nxcl(m,I),I)) Then ! Extrapolate towards x=mu
                  If(CLmumu(I,m).lt. .995) Then
C Extrapolate from two points, the largest tabulated xcl and mu.  Use another
C type of exponential extrapolation.
                     x0=xcl(m,Nxcl(m,I),I)
                     x1=mu
                     p0=1.-CL(Nxcl(m,I))
                     p1=1.-CLmumu(I,m)
                  Else
C Extrapolate using as the two points the largest two xcl's
                    x0=xcl(m,Nxcl(m,I)-1,I)
                    x1=xcl(m,Nxcl(m,I),I)
                    p0=1.-CL(Nxcl(m,I)-1)
                    p1=1.-CL(Nxcl(m,I))
                 EndIf
                 b=log(p0/p1)
C In either case, p0 > p1 and x1 > x0, so b>0
C Choose a form that's exact at the two points used, must increase with x,
C and must not be >1, and make it a form that has been checked to work ok
C for nearby extrapolation.
                 p(J)=1.-p0*exp((-b)*(x-x0)/(x1-x0))
C If Nxcl was 1 then either x >= only xcl or x <= only xcl, and has already
C been handled.  So from now on, Nxcl .ge. 2 and x is between the lowest and
C highest tabulated xcl.
               Else ! Do quadratic interpolation in x.
                  If(Nxcl(m,I).eq. 2) Then
C interpolate quadratically between the two tabulated values with the
C third point taken to be at x=mean(I)
                     x0=xcl(m,1,I)
                     x1=xcl(m,2,I)
                     x2=mean(I)
                     p0=CL(1)
                     p1=CL(2)
                     p2=CLmumu(I,m)
                  Else
C From now on, Nxcl .ge. 3 and x is between the lowest and highest MC values.
C Use a binary search to find the first xcl which is at least x, xcl(m,K,I).
C Then quadratically interpolate between K, K+1, and K+2.
                     KLow=1 ! Eventually K will become KLow
                     KHigh=Nxcl(m,I)-1  ! KHigh starts at least 2
 40                  Continue
                     If(KHigh.le.KLow+1) Then ! Done with the search
                        K=KLow
                        Go to 50
                     EndIf
C KLow and KHigh are separated by at least 2.
                     KMid=(KLow+KHigh)/2
                     If(x.gt.xcl(m,KMid,I)) Then
                        KLow=KMid
                     Else
                        KHigh=KMid
                     EndIf
                     Go to 40
 50                  Continue
                     x0=xcl(m,K,I)
                     x1=xcl(m,K+1,I)
                     x2=xcl(m,K+2,I)
                     p0=CL(K)
                     p1=CL(K+1)
                     p2=CL(K+2)
                  EndIf ! Finished finding quadratic interpolation points
                  d0=(x0-x1)*(x0-x2)
                  d1=(x1-x0)*(x1-x2)
                  d2=(x2-x0)*(x2-x1)
C p(J) is the quadratic form that's the correct CL at that I for x0,
C x1, and x2, and is evaluated at x.
                  p(J)=p0*(x-x1)*(x-x2)/d0 + p1*(x-x0)*(x-x2)/d1
     1                +p2*(x-x0)*(x-x1)/d2
C If the quadratic form works relatively poorly, it can put p(J) outside
C its possible range.  This can happen, but let it be, with just a warning,
C which is dropped if p(J) is close enough to its range
                  If(p(J) .lt. -2.e-6 .or. p(J)
     1                .gt. CLmumu(I,m)+.00003) Then
C Don't worry if p(J) is just a little out of range
                    Icode=OR(Icode,1)
                  EndIf
C Done with quadratic interpolation in x
               EndIf  ! Finished with getting p(J) for this mean
            EndDo  ! End loop over J <--> neighboring means.
C Do quadratic interpolation in mean unless x >= mean(I0+1), when linear.
C Normally mean(I0)<mu and mean(I0+1)>mu, but this isn't true if I0+2 > Nmeans,
C in which case x can be > mean(I0+1) and still not be > mu, which has already
C been handled.
            If(x.ge.mean(I0+1)) Then
               ConfLev=p(1)+(mu-mu0(1))*(p(2)-p(1))/(mu0(2)-mu0(1))
            Else
               d0=(mu0(0)-mu0(1))*(mu0(0)-mu0(2))
               d1=(mu0(1)-mu0(0))*(mu0(1)-mu0(2))
               d2=(mu0(2)-mu0(0))*(mu0(2)-mu0(1))
               ConfLev=p(0)*(mu-mu0(1))*(mu-mu0(2))/d0 +
     1             p(1)*(mu-mu0(0))*(mu-mu0(2))/d1 +
     2             p(2)*(mu-mu0(0))*(mu-mu0(1))/d2
            EndIf
         EndIf ! End of the case of at least two non-zero Nxcl's.
      EndIf ! End of the case in which the table can include the m and mu.
C Prevent tiny, but not big, differences below 0 or above 1.
      If(ConfLev.lt.0. .and. ConfLev.gt. -1.E-6) ConfLev=0.
      x0=1. - ConfLev
      If(x0.lt.0. .and. x0.gt. -1.E-6) ConfLev=1.
      If(ConfLev.lt.0. .or. ConfLev.gt. 1.) icode=1
      return
 1000 ConfLev=-1.
C Should never have been called with the given parameters
      icode=2
      return
      end
