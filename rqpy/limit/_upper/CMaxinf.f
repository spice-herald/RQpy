      Real Function CMaxinf(CL,Ifmin,mu)
C Uses table CMaxf.in, which was made by CMaxinfgen, to compute
C the CL confidence level value of CMaxinf for minimum f of the
C Ifmin value of fmin and for total expected number of events, mu.
C fmins has the list of values of Nfmin values of fmin.
C As of this writing, correspondence between Ifminin and fmin:
C Ifmin  fmin
C   1    0.00
C   2    0.01
C   3    0.02
C   4    0.05
C   5    0.10
C   6    0.20
C   7    0.50 = fmins(Nfmin)
C
C As of this writing, CLs consist of NCLs=40 values from .8 to .995.
C CL is not allowed to be less than CLs(1) or greater than CLs(NCls).
C If mu is too low, we can expect the result to be unreliable, but it's
C permitted.  If mu > mus(Nmus), give the same result as mu=mus(Nmus)
C or, if extrapolation is available, use A + B/sqrt(mu).  If mu<mus(Nmus),
C linearly interpolate in log(mus),CLs.
C
C Feb. 2011: Use CMax.txt for mu < 99.4843, including very low mu and CL.
C Before trying to interpolate in CL3s, check if C_0(mu,mu) is above CL and
C C_1(mu,mu) is below CL.  If these conditions are met, CMaxinf2=CL.  If
C C_1(mu,mu) is above CL, find what CMaxinf2 will be for the given mu for
C each of the CL3s on either side of CL, then interpolate.  For each CL3s:
C For mu just after mu0, where C_0(mu0,mu0)=CL3s, until mu1, where
C C_1(mu1,mu1)=CL3s, the value of CMaxinf=CL3s.  For mu just after mu1,
C CMaxinf=C_1(mu1,mu1).  There are also ranges of mu > mun, depending on CL3s,
C where CMaxinf=C_n(mun,mun).
      Implicit None
      Real CL,mu,Fmu,CMaxCL,CMaxIn
      External Fmu
      Integer Ifmin,Nmus,NmuMax,NCLs,NCLmax,Nfmin,NfminMax,I,J,K,L,
     1 Imu,ICL,Imu24,Imu14,Imu34,Imustart,NmuMax2,
     2 NCL3s,NmuMax3,Nmus3,N,NCL3sa,NTrials,ICLow,ICHigh,ICMid
      Parameter (NCLmax=40)
      Parameter (NCL3s=84) ! For CMax.txt
      Parameter (NfminMax=7)
      Parameter (NmuMax=50)
      Parameter (NmuMax2=730) ! 723 should be needed
      Parameter (NmuMax3=891) ! L=-430 to +460 in CMax.txt mu=exp(.01*L)
      Real CLs(NCLmax),mus(NmuMax),Table(NCLmax,NfminMax,NmuMax),
     1 fmins(NfminMax),logmu(NmuMax),lmu,xCL,xmu,sqrtmu,
     2 A(NClmax,NfminMax),B(NClmax,NfminMax),CL3s(NCL3s),
     4 mucut3/99.4843/,mus3(NmuMax3),Table3(NCL3s,NmuMax3),
     5 expmu,mu0,BotI(NCL3s),TopI(NCL3s),CLTop(NCL3s),CLext(0:1),
     6 CMax(0:1),sigcut,RRR,eps/1.E-6/,ACL(NmuMax3),BCL(NmuMax3)
      Integer Nstats(NmuMax3)
      Logical first/.true./,UseTable3/.true./,InvFun/.false./,
     1 NewGroup/.true./
      Common/CMaxinfcom2/NCLs,Nfmin,CLs,fmins,mus,Table,Nmus,Imu24,
     1 Imu14,Imu34,logmu,A,B,CL3s,nmus3,mus3,table3,BotI,TopI,CLTop,
     2 ACL,BCL
      Integer IType,NCalls
      Real x0,x1,y0,y1
      Common /FMucom/IType,NCalls,x0,x1,y0,y1
 2    If(first) Then
         nmus3=NmuMax3
         first=.false.
C CMaxf.txt has the high statistics data limit from mu=15 to 15310,
C along with information needed for extrapolating higher.
         Open(21,file='CMaxf.txt',status='OLD',form='FORMATTED')
         Read(21,4) NCLs,Nfmin
 4       Format(2I4)
         Read(21,6) (CLs(I),I=1,NCLs),(fmins(I),I=1,Nfmin)
 6       Format(47F6.3)
         Nmus=0
         A(1,1)=0. ! A(1,1)=0 implies extrapolation data is unavailable
 50      Continue
         Nmus=Nmus+1
         read(21,55,End=100) mus(Nmus)
 55      Format(F9.3)
         If(mus(Nmus).eq.0.) Then
C mus(Nmus)=0 ==> no more Table entries but maybe extrapolation data.
             Read(21,60) ((A(ICL,J),B(ICL,J),ICL=1,NCLs),
     1       J=1,Nfmin)
 60         Format(10F9.5)
            Go to 100
         EndIf
         logmu(Nmus)=log(mus(Nmus))
         Do J=1,Nfmin
           Read(21,60) (Table(I,J,Nmus),I=1,NCLs)
         EndDo
         Goto 50
 100     Close(21)
         Nmus=Nmus-1
         Imu24=Nmus/2
         Imu14=Imu24/2
         Imu34=3*Nmus/4
         If(UseTable3) Then
C CMax.txt has low statistics information from mu=.0136 to 99.48
C corresponding to mus3(1)=exp(-4.30) to mus3(891)=exp(+4.60) in steps
C of exp(.01).  I.e., mus3(I) should be exp(.01*(I-431)).  CL3s(I)
C run from .0001 to .9999 in irregular steps.  If CL < .0001 doesn't matter,
C then for mu<.01 only C0 matters, and CMax=C0(mu).
           Open(21,file='CMax.txt',status='OLD',
     1        form='FORMATTED')
           Do I=1,Nmus3
 104        Continue
            If(NewGroup) Then
              NewGroup=.false.
              Read(21,106) NTrials,N,NCL3sa
 106          Format(I9,2I4)
              If(NCL3sa.ne.NCL3s .or. N.ne.100) Then
                Write(6,*) "Bad CMax.txt, with NCL3sa, N: ",NCL3sa,N
                Stop
              EndIf
              Read(21,108) CL3s
 108          Format(10F8.5)
            EndIf
            Read(21,110) mus3(I)
 110        Format(F12.5)
            If(mus3(I).gt.0.) Then
               Nstats(I)=NTrials
               Read(21,108) (Table3(J,I),J=1,NCL3s)
            Else
               NewGroup=.true.
               Go to 104
            EndIf
           EndDo
           Close(21)
           mucut3=0.999999*mus3(nmus3)
C Find for each mus3 the ACL and BCL for which at low CMax we have
C CL=(ACL*CMAX)**BCL.  These numbers are used by CMaxCL, although it's
C probably a poor approximation for CL3s(2) less than 1-(1+mu)*exp(-mu).
C
           y0=log(CL3s(2)/CL3s(1))
           Do J=1,nmus3
              BCL(J)=log(Table3(2,J)/Table3(1,J))/y0
              ACL(J)=(CL3s(1)**BCL(J))/Table3(1,J)
              BCL(J)=1./BCL(J)
           EndDo
C
C Find for each CL3s(J) the xI in mu=mus3(xI) for which C_1(mu,mu)
C = CL3s(J).   First find the mu, then take (J) = 431.+100.*log(mu).
C This is the mu at which CMax starts to quickly rise from its previously
C flat value of CL3s(J).
           Do J=1,NCL3s
C Take CMaxinf = C_1(mu,mu) from mu1, where C_1(mu1,mu1)=CL, and mu2, found
C as follows:  Step along mu=mus3(K) until the end of the C_1 region.  Solve
C C_1(mu2,mu2) = extrapolation of the straight line I-fit of the tabulated CMax
C at K and K+1.  Save the effective I of mu1 and mu2, and save C_1(mu2,mu2).
C Usage:
C If mu2 < mu < mus3(I) for the first I beyond mu2 (the smallest integer .ge.
C the effective I of mu2), interpolate between mu2 and mus3(I).  Otherwise
C linearly interpolate between mus3 on either side of mu.
C Perhaps this should be done only for .001 .le. CL .le. .999, but tabulate
C .0005 .le. CL .le. .9995.  This idea can be extended to getting data for
C other jumps corresponding to other C_n(mu,mu).
              y0=CL3s(J)
              Itype=0
              Call RZERO(mus3(1),20.,mu0,RRR,eps,20,Fmu)
              BotI(J)=431.+100.*log(mu0)
              If(CL3s(J) .lt. .00049 .or. CL3s(J) .gt. .99951) Go to 112
C Don't do this for extreme CL
              Itype=1
              I=BotI(J)+1
              Do K=I,I+100
C The end of the CMax=C_1(mu,mu) region is detected by finding that
C the tabulated MC value of CMax + 4*sigma+.00002 is < C_1, where
C sigma = sqrt(CL*(1-CL)/NTrials), and for mu<15 NTrials=1.E8.
               sigcut=4.*sqrt(y0*(1.-y0)/float(Nstats(K)))+0.00003
               If( Table3(J,K)+sigcut .lt. 
     1          1.-(1.+mus3(K))*exp(-mus3(K)) ) Then
                 x0=float(K)
                 x1=x0+2.
                 y0=Table3(J,K)
                 y1=Table3(J,K+2)
                 Call RZERO(mus3(1),20.,mu0,RRR,eps,20,Fmu)
                 TopI(J)=431.+100.*log(mu0)
                 CLtop(J)=1.-(1.+mu0)*exp(-mu0)
                 If(max(Table3(J,K-1),y0,y1).gt.CLtop(J)) Then
C If the extrapolation gets CLtop below the highest nearby tabulated value, use
C the highest nearby tabulated value.
                    If(Table3(J,K-1).ge.Max(y0,y1)) Then
                       L=K-1
                    ElseIf( y0.ge.Max(Table3(J,K-1),y1) ) Then
                       L=K
                    Else
                       L=K+1
                    EndIf
                    CLtop(J)=Table3(J,L)
                    y0=CLtop(J)
                    Itype=0
                    Call RZERO(mus3(1),20.,mu0,RRR,eps,20,Fmu)
                    TopI(J)=431.+100.*log(mu0)
                 EndIf
                 Go to 112
               EndIf
              EndDo
              Write(6,*) "CMaxinfNew failed to find the top",J
              stop
 112          Continue
           EndDo
         EndIf
         If(InvFun) Go to 500
      EndIf ! End of table readin
C
      If(Ifmin.gt.Nfmin .or. Ifmin.lt.1) Then
          Write(6,115) Ifmin
 115      Format(I9,' is outside the allowed range of Ifmin values')
          Stop
      EndIf
      If(UseTable3 .and. mu.le.mucut3) Then
        expmu=exp(-mu)
        If(CL.gt. 1.-expmu) Then
            CMaxinf=2. ! This mu cannot be rejected to CL by any observation
        ElseIf(CL.gt.1.-expmu*(1.+mu) .or.mu.le.CL3s(1)) Then
C For small enough mu I assume C0 almost always is the optimum interval.  Also
C if C1(mu,mu) < CL <  C0(mu,mu) then only C0 can give an interval with that
C CL, in which case its confidence level is the overall one.
            CMaxinf=CL
        ElseIf(CL.ge.CL3s(NCL3s)) Then
            CMaxinf=1.
        Else
C Find the ICL for which CL3s(ICL) .lt. CL .le. CL3s(ICL+1) by binary search.
C If CL < CL3s(1) then ICL=1 and if CL > CL3s(NCL3s) then ICL=NCL3s-1, but
C I've already taken care of these possibilities.
         ICLow=1
         ICHigh=NCL3s
 117     Continue
         If(ICHigh.le.ICLow+1) Then ! Done with the search
            ICL=ICLow
            Go to 118
         EndIf
C ICLow and ICHigh are separated by at least 2.
         ICMid=(ICLow+ICHigh)/2
         If(CL.gt.CL3s(ICMid)) Then
            ICLow=ICMid
         Else
            ICHigh=ICMid
         EndIf
         Go to 117
 118     Continue
C
C For each of ICL and ICL+1 linearly interpolate in log(mus3) to get for that
C ICL the CMax.  Then interpolate in CL.  If for a CL3s mu is just before the
C start of the rise or just after the end, one of the points used will be at
C the start or end of the rise.  If mu is between the start and end of the
C rise, use C_1(mus3,mus3).
         xmu=431. + 100.*log(mu)
         Imu=xmu
C         xmu=xmu-float(Imu) ! Now xmu is the fraction of the way to Imu+1
         xCL=(CL-CL3s(ICL))/(CL3s(ICL+1)-CL3s(ICL)) ! fraction of way to ICL+1
C Interpolate Table with CL between CL3s(ICL) and CL3s(ICL+1) and with
C mu between mus3(Imu) and mus3(Imu+1).
         If(Imu.ge.NmuMax3 .or. Imu.lt.1) Then
            Write(6,*) "CMaxinfNew finds illegal Imu:",Imu
            Stop
         EndIf
         Do J=0,1
            I=ICL+J
            K=TopI(I)
            If(xmu.le.BotI(I)) Then
               CMax(J)=CL3s(I)
               Go to 130
            ElseIf(xmu.ge.K+1 .or. I.lt.3 .or. I.gt. 82) Then
               If(I.lt.1 .or. I.gt. NCL3s) Then
                  Write(6,*) "CMaxinfNew finds illegal ICL:",ICL
                  Stop
               EndIf
               x0=float(Imu)
               x1=x0+1.
               y0=Table3(I,Imu)
               y1=Table3(I,Imu+1)
            ElseIf(xmu.ge.TopI(I)) Then
C xmu must be between TopI(I) and Imu+1, and TopI(I) is .ge. Imu
               x0=TopI(I)
               x1=float(Imu+1)
               y0=CLTop(I)
               y1=Table3(I,Imu+1)
            ElseIf(xmu.ge.BotI(I)) Then
               CMax(J)=1.-(1.+mu)*exp(-mu)
               Go to 130
            EndIf
            CMax(J)=y0 + (xmu-x0)*(y1-y0)/(x1-x0)
 130        Continue
         EndDo
         CMaxinf=CMax(0) + xcl*(CMax(1)-CMax(0))
        EndIf
        Goto 300
      EndIf
C Find between what CL's we should interpolate
C Here we assume 40 CL's from .8 to .995.
      ICL=(CL-.795)*200.
C Allow CL as low as .795 with extrapolation.
      If(ICL.eq.0) ICL=1
      If(ICL.lt.1 .or. ICL.ge.NCLs) Then
          Write(6,160) CL
 160      Format(F9.4,' is outside the allowed range of CL values')
          CMaxinf=-1.
          Return
      EndIF
C To speed this up, first find what quarter of the range should
C be searched.  Then for simplicity over a complete binary search,
C complete the search in a dumb way.
      If(mus(Imu24).ge.mu) Then
         If(mus(Imu14).ge.mu) Then
            Imustart=2
         Else
            Imustart=Imu14
         EndIf
      Elseif(mus(Imu34).ge.mu) Then
         Imustart=Imu24
      Else
         Imustart=Imu34
      EndIf
      Do I=Imustart,Nmus
          Imu=I
          If(mus(I).ge.mu) Then
             Go to 200
          EndIf
      EndDo
C mu is greater than all the tabulated values; so use the top one or
C use A and B
 200  Continue
      lmu=log(mu)
      xCL=(CL-CLS(ICL))/(CLS(ICL+1)-CLS(ICL))
      If(mu.gt.mus(Nmus) .and. A(1,1).ne.0.) Then
C Interpolate the computed extrapolated value between CLs(ICL) and
C CLs(ICL+1).
       sqrtmu=sqrt(mu)
       CMaxinf=(1.-xCL)*(A(ICL,Ifmin) + B(ICL,Ifmin)/sqrtmu) +
     1   xCL*(A(ICL+1,Ifmin) + B(ICL+1,Ifmin)/sqrtmu)
      Else
        Imu=Imu-1
C Interpolate Table with CL between CLs(ICL) and CLs(ICL+1) and with
C lmu between logmu(Imu) and logmu(Imu+1).
        xmu=(lmu-logmu(Imu))/(logmu(Imu+1)-logmu(Imu))
         CMaxinf=(1.-xCL)*((1.-xmu)*Table(ICL,Ifmin,Imu) +
     1   xmu*Table(ICL,Ifmin,Imu+1)) +
     2   xCL*((1.-xmu)*Table(ICL+1,Ifmin,Imu) +
     3   xmu*Table(ICL+1,Ifmin,Imu+1))
      EndIf
 300  Continue ! CMaxinf=max(CMaxinf,CL)
C The above statement may be needed if mu is too low for the table.
C If the table is made with steps in mu of, say 0.025, then between
C the threshold of where CMaxinf should be defined and where the
C first non-zero entry of the table is there should be a legitimate
C return of CL.
      return
      Entry CMaxCL(CMaxIn,Ifmin,mu)
C Computes the inverse of CMaxinf: CMaxCL(CMaxInf(CL,Ifmin,mu),Ifmin,mu)=CL
C For now, only for mu < mucut3, assume fmin=0.  Linearly interpolate between
C the two neighboring CL and the two neighboring mu.  CL<.0001 or mu < mus3(1)
C => take CMaxCL=CMaxIn.  Unfortunately there are ranges of CL which give the
C same CMaxInf (at the rapid rises in CMaxInf), so the inverse has a range.
      If(first) Then
         InvFun=.true.
         Go to 2
      EndIf
 500  Continue
      expmu=exp(-mu)
      CMaxCL=CMaxIn
      If(CMaxIn.gt. 0.99999 .or. CMaxIn .le. 0. .or. mu.gt.mucut3)
     1  Then
         Write(6,*) "CMaxCL unable to handle CMaxIn, mu = ",CMaxIn,mu
      ElseIf(mu.gt.mus3(1) .and. 1.-(1.+mu)*expmu .ge. CMAXIn ) Then
         lmu=log(mu)
         xmu=431. + 100.*lmu
         Imu=xmu
         Do J=0,1
            I=Imu+J
            If(CMaxIn.le.Table3(2,I)) Then
C Extrapolate or interpolate below the table entry for CLs3(2).  The following
C line gives the exact result for CMaxIn=Table3(1,I) and Table3(2,I).
               CLext(J)=(ACL(I)*CMaxIn)**BCL(I)
            Else
C Do a binary search for the CL3s between which to do a linear interpolation.
               ICLow=1
               ICHigh=NCL3s
 517           Continue
               If(ICHigh.le.ICLow+1) Then ! Done with the search
                  ICL=ICLow
                  Go to 518
               EndIf
C ICLow and ICHigh are separated by at least 2.
               ICMid=(ICLow+ICHigh)/2
               If(CMaxIn.gt.Table3(ICMid,I)) Then
                  ICLow=ICMid
               Else
                 ICHigh=ICMid
               EndIf
               Go to 517
 518           Continue
C Linearly interpolate between CL3s(ICL) and CL3s(ICL+1).  One may verify
C that CMaxIn=Table3(ICL,I) => CLext=CL3s(ICL) and CMaxIn=Table3(ICL+1,I)
C => CLext=CL3s(ICL+1).
               xCL=(CMaxIn-Table3(ICL,I))/
     1            (Table3(ICL+1,I)-Table3(ICL,I))
               CLext(J)=CL3s(ICL)+xCL*(CL3s(ICL+1)-CL3s(ICL))
            EndIf
         EndDo
C Interpolate between the two adjacent mu values with J=0 and J=1
         CMaxCL=CLext(0)+(xmu-float(Imu))*(CLext(1)-CLext(0))
      EndIf
      Return
      End
C
      Real Function FMu(mu,I)
      Implicit None
      Integer I
      Real mu
      Integer IType,NCalls
      Real x0,x1,y0,y1
      Common /FMucom/IType,NCalls,x0,x1,y0,y1
      Real mulog
      If(I.eq.1) NCalls=0
      FMu=1.-(1.+mu)*exp(-mu) ! First set FMu to C_1(mu,mu)
      If(IType.eq.0) Then
C Find where C_1(mu,mu) equals a particular value, y0
         FMu=y0-FMu
      ElseIf(Itype.eq.1) Then
C Find where C_1(mu,mu) crosses a particular straight line
         mulog=431.+100.*log(mu)
         FMu=y0+(mulog-x0)*(y1-y0)/(x1-x0) - FMu
      EndIf
      NCalls=NCalls+1
      Return
      End
