      Real Function Upper(Method,CL,Nexp,Maxp1,Nevts,mu,FC,ICode)
C Calculates the upper limit cross section for a combination of experiments
C
C Method: Method as in CDMS_Edelweiss.pdf.  Must be -6 to +6 but not -4 or 0.
C         Positive values use the optimum interval; negative use maximum gap.
C Method  Nexp     Name
C     0    1    Simple Merging (assumes merging already done, so = Upperlim)
C     1    >0   Merging after transforming to cumulative probabilities
C     2    >0   Serialization
C     3    1-2  Summed Gap
C     4    >0   Minimum Limit
C     5    2    Minimum Probability
C     6    2    Probability Product
C  Negative Same as positive, but uses maximum gap, not optimum interval
C         
C CL:     The desired confidence level ( 0 < CL < 1 ).
C Nexp:   The number of experiments to be combined.  For Method 0 Nexp must
C         be 1.  For Method 1, 2, and 4 it must be 1-20.  For Method 3
C         2-20.  For Method 5, and 6 it must be 2.  For Method 3 it must be 
C Maxp1:  At least the largest number of events in the experiments plus 1
C Nevts:  Integer array Nevts(Nexp) = number of events in each experiment.
C         Nevts(Nexp) must be non-negative.
C mu:     Real array mu(Nexp) = expected number of events per unit
C         cross section (you choose the "unit") in each experiment.
C         mu(Nexp) must be > 0.
C FC:     Real array FC(0:Maxp1,Nexp).  Given the foreground distribution
C         whose shape, but not normalization, is known for each of Nexp
C         experiments, for event I of experiment J, FC(I,J) = probability
C         of an event in the distribution being earlier in the measurement
C         range, with events ordered from earlier to later in the range.
C         FC(0,J) will be set to 0.0 and FC(Nevts(J)+1,J) will be set to 1.0.
C ICode:  Return code
C    0    no problem encountered.
C    1    If Method=3, the range investigated had to be expanded once
C    2    If Method=3, the range investigated had to be expanded twice.  That
C         is ok if RZERO didn't print out an error message.
C 0-511   If Method=1, 2, 4, 5, this is the UpperLim return code.
C         If Method=3, this is the last code return from CombConf
C 512     For Method 3 RZERO fit failed.  For Method 4 no solution with CL
C         so found a CLNow>CL (in common UpperLimCom) with a solution.
C 1024    Method, CL, Nexp, Nevts, mu, or FC not allowed
C 2048    For Method=4 with no solution at CL there's also no solution found.
      Implicit None
      Integer Method,Nexp,Maxp1,Nevts(Nexp),ICode,NTot,Maxevts,NLowmu,
     1 LowExp
      Parameter (Maxevts=20000)
      Real CL,mu(Nexp),FC(0:Maxp1,Nexp),FC1(0:Maxevts),UpperLim,
     1 EPS/.00001/,Fupper,R,mutot,BotSigma,TopSigma,sigma,mucut,
     2 SigmaLow,SigmaPrev,DelSigma,F1,F2,absBot
      Real CL0,mu0(20),xmax(20),f(0:Maxevts,2),p1,p2
      Integer Method0,Nexp0,IFlag0,Nevts0(2),Icode1,Icode2
      Common/UpperCom/Method0,CL0,Nexp0,xmax,mu0,IFlag0,f,Nevts0,
     1 p1,p2
      Real Exclude_Low,CLNow
      Integer EndPoints
      Common/UpperLimcom/EndPoints(2),Exclude_low(2),CLNow
      External Fupper
      Integer I,J,K,MAXF/50/,Nexpand,IFlag,NSearch,NSearchMax
      Logical BinSearch,FoundOne
      Real CLLow,CLHigh
      Upper=0.
      ICode=1024
      If(CL.le.0. .or. CL.ge.1.) Then
         Write(6,*) "CL cannot be",CL
         Return
      ElseIf(Nexp.lt.1 .or. (Method.eq.0 .and. Nexp.ne.1) .or.
     1 (Method.eq.3 .and. Nexp.gt.2) .or. (Method.gt.4 .and.
     2 Nexp.ne.2)) Then
         Write(6,*) "Nexp cannot be", Nexp
         Return
      ElseIf(abs(Method).lt.0 .or. abs(Method).gt.6 .or.
     1   Method.eq.-4) Then
         Write(6,*) "Method cannot be",Method
         Return
      Else
         Do I=1,Nexp
            If(mu(I).le.0.) Then
               Write(6,*) "mu cannot be",mu(I)
               Return
            ElseIf(Nevts(I).lt.0) Then
               Write(6,*) "Nevts cannot be",Nevts(I)
            EndIf
          EndDo
      EndIf
      Icode=0
C
      If(Method.eq.0) Then
         If(Nexp.ne.1) Then
            Write(6,*) "Method ",Method," must have Nexp=1, not",Nexp
            Icode=1024
            return
         EndIf
         Upper=UpperLim(CL,1,Nevts(1),FC(0,1),0,0,ICode)/mu(1)
C
      ElseIf(Method.eq.1) Then
         NTot=0
         mutot=0.
         FC1(0)=0.
         Do I=1,Nexp
            If(Nevts(I).gt.0) Then
               Do J=1,Nevts(I)
                  FC1(NTot+J)=FC(J,I)
               EndDo
               NTot=NTot+Nevts(I)
               mutot=mutot+mu(I)
               If(I.lt.Nexp) Then
                  If (NTot+Nevts(I+1).ge.Maxevts) Then
                    ICode=1024
                    return
                  EndIf
               EndIf
            EndIf
         EndDo
C NTot events in FC1(1) to FC1(NTot) + 1 for FC1(0) are to be sorted:
         Call SORTR(FC1,1,NTot+1,1)
         Upper=UpperLim(CL,1,NTot,FC1,0,0,ICode)/mutot
C
      ElseIf(Method.eq.2) Then
         NTot=0
         mutot=0.
         Do I=1,Nexp
            If(Nevts(I).gt.0) Then
               Do J=1,Nevts(I)
                  FC1(NTot+J)=mu(I)*FC(J,I)+mutot
               EndDo
            EndIf
            NTot=NTot+Nevts(I)
            mutot=mutot+mu(I)
            If(I.lt.Nexp) Then
             If (NTot+Nevts(I+1).ge.Maxevts) Then
               Write(6,*) "Too many events.  Must be fewer than",Maxevts
               ICode=1024
               return
             EndIf
            EndIf
         EndDo
         Do I=1,NTot
            FC1(I)=FC1(I)/mutot
         EndDo
         Upper=UpperLim(CL,1,NTot,FC1,0,0,ICode)/mutot
C
      ElseIf(Method.eq.4) Then
C Do a binary search for a CLnow > CL if no solution at CL with NSearchMax
C cuts in half of the search region.
         Binsearch=.false.
         FoundOne=.false.
         NSearchMax=12
         CLnow=CL
C Loop to 90 if doing a binary search for CLNow
 90      K=Nexp
C        SigmaPrev=1.E30
 100     Continue
         CL0=exp(log(CLNow)/Float(K))
         If(CL0.gt. .995) Then
            Write(6,*) CL," is too high for ",Nexp," experiments."
            ICode=1024
            Return
         EndIf
         SigmaLow=1.E30
         LowExp=0
         ICode=0
         Do I=1,Nexp
            If(Nevts(I).eq.0) Then
               Sigma=-log(1.-CL0)
            Else
               Sigma=UpperLim(CL0,1,Nevts(I),FC(0,I),0,0,Iflag)
               ICode=Or(ICode,Iflag)
C For Method 4, ICode includes any bit set by any call to UpperLim.
            EndIf
            Sigma=Sigma/mu(I)
            If(Sigma.lt.SigmaLow) Then
               SigmaLow=Sigma
               LowExp=I
            EndIf
         EndDo
         mucut=-log(1.-CL0)
         NLowmu=1
         Do I=1,Nexp
           If(I.ne.LowExp .and. mu(I)*SigmaLow.ge.mucut)
     1        NLowmu=NLowmu+1
         EndDo
         If(NLowmu.lt.K) Then
C            SigmaPrev=SigmaLow
            K=K-1
            Go to 100
         ElseIf(NLowmu.gt.K) Then
            If(BinSearch) Then
               CLLow=CLNow
               CLNow=.5*(CLLow+CLHigh)
            Else
C Start a binary search for the lowest CLNow > CL which gives NLowmu=K
               BinSearch=.true.
               NSearch=0
               CLLow=CL
               CLHigh=CL**(Float(K)/Float(NLowmu))
               CLNow=.5*(CLLow+CLHigh)
            EndIf
            Go to 200
C            SigmaLow=SigmaPrev
         Else
            If(BinSearch) Then
               FoundOne=.true.
               CLHigh=CLNow
               CLNow=.5*(CLLow+CLHigh)
               Go to 200
            Else
               Upper=SigmaLow
               Return
            EndIf
         EndIf
 200     Continue
         If(NSearch.ge.NSearchMax) Then
            Upper=SigmaLow
            ICode=Or(ICode,512)
            If(.not.FoundOne) ICode=Or(ICode,2048)
         Else
            NSearch=NSearch+1
            Go to 90
         EndIf
C
      Else
C abs(Method) equals 3, 5, or 6 or Method = -1 or -2
C Renormalize the internal "unit" cross section so that the total number of
C expected events per new "unit" cross section is unity.
         Method0=Method
         mutot=0.
         Do I=1,Nexp
          mutot=mutot+mu(I)
          Nevts0(I)=Nevts(I)
         EndDo
         Do I=1,Nexp
           mu0(I)=mu(I)/mutot
         EndDo
C
         Nexp0=Nexp
         CL0=CL
         If(Nexp.gt.2) Then
            Icode=1024
            Write(6,*) "Nexp cannot be",Nexp
            Return
         EndIf
         NTot=0
         Do I=1,Nexp
           NTot=NTot+Nevts(I)
           FC(0,I)=0.
           FC(Nevts(I)+1,I)=1.
           If(Method.eq.-3) Then
             Call ForCnMax(Nevts(I),FC(0,I),f(0,I),0)
             xmax(I)=f(0,I)
C             xmax(I)=0.
C             Do J=0,Nevts(I)
C               xmax(I)=max(xmax(I),FC(J+1,I)-FC(J,I))
C For (renormalized) cross section sigma, the maximum gap of experiment I
C is mu0(I)*sigma*xmax(I).
C             EndDo
           Else
             Call ForCnMax(Nevts(I),FC(0,I),f(0,I),Nevts(I))
           EndIf
         EndDo
         absBot=-log(1.-CL)
         If(.not. (absBot .gt. 0. .and. absbot .lt. 100.)) Then
C            Write(6,*) "Upper miscalculates -log(1-CL)",absBot
            absBot=-log(1.-CL)
C            Write(6,*) "Tried again:", absBot
         EndIf
         If(NTot.eq.0) Then
            sigma=absBot
         Else
C Empirically I expect the resulting sigma upper limit to be roughly
C 1.3*NTot to within a few times sqrt(NTOT+5.).
            TopSigma=(1.4+2.*(CL-.9))*Float(NTot)
            DelSigma=8.*sqrt(Float(NTot)+5.)
            BotSigma=Max(absBot,TopSigma - DelSigma)
            TopSigma=TopSigma+DelSigma
C Count the number of times the range searched by RZERO must be expanded
            Nexpand=0
C If it's > 1, give a warning.  Expand it at most twice
 50         F1=Fupper(BotSigma,0)
            F2=Fupper(TopSigma,0)
            If(F1*F2 .gt. 0.) Then
               Nexpand=Nexpand+1
               If(F1.gt.0.) BotSigma=Max(absBot,BotSigma-DelSigma)
               If(F2.lt.0.) TopSigma=TopSigma+DelSigma
               If(Nexpand.gt.1) Then
                Write(6,*) "Inadequate initial range for Method 3:",
     1           BotSigma,TopSigma,F1,F2
C If this is the last try, make BotSigma as low as it can be (0 events)
C and increase TopSigma by (Nexpand+2)*DelSigma above its original value.
                 BotSigma=absBot
                 TopSigma=TopSigma+DelSigma
               Else
                  Go to 50
               EndIf
            EndIf
C            TopSigma=2.*(float(NTot+5) + 4.*sqrt(Float(NTot)))
            Call RZERO(BotSigma,TopSigma,sigma,R,EPS,MAXF,Fupper)
            If(R.lt.0.) Then
               Write(6,*) "RZERO returns R", R
               Write(6,*)" Fupper(BotSigma) =",Fupper(BotSigma,0),
     1        ", Fupper(TopSigma) =",Fupper(TopSigma,0)
               ICode=512
            EndIf
         EndIf
         ICode=IFlag0
         Upper=sigma/mutot
C
      EndIf
      Return
      End
      Real Function Fupper(sigma,I)
      Implicit None
      Real sigma
      Integer I,Maxevts,mtop,Istat
      Real C0,CombConf,mu1,mu2,CnMax,CMaxCL,CL
      Parameter (Maxevts=20000)
      Real CL0,mu0(20),xmax(20),f(0:Maxevts,2),p1,p2
      Integer Method0,Nexp0,IFlag0,Nevts0(2)
      Common/UpperCom/Method0,CL0,Nexp0,xmax,mu0,IFlag0,f,Nevts0,
     1 p1,p2
      Fupper=0.
      If(abs(Method0).gt.2 .or. Method0.eq.-1) Then
        mu1=sigma*mu0(1)
        If(Method0.lt.0) Then
           p1=C0(mu1*xmax(1),mu1)
        Else
           CL=CnMax(Nevts0(1),f(0,1),mu1,99.,0.,mtop,Istat)
           If(Istat.gt.0) Then
              Write(6,*) "Fupper CnMax returned error ",Istat,
     1      " for Method ",Method0," experiment 1"
           EndIf
           If(CL.lt. 0.99999 .and. mu1 .lt. 99.) Then
              p1=CMaxCL(CL,0.,mu1)
           Else
              p1=1.
           EndIf
        EndIf
        If(Nexp0.eq.1) Then
           Fupper=p1-CL0
        Else
           mu2=sigma*mu0(2)
           If(Method0.lt.0) Then
             p2=C0(mu2*xmax(2),mu2)
           Else
             CL=CnMax(Nevts0(2),f(0,2),mu2,99.,0.,mtop,Istat)
             If(Istat.gt.0) Then
                Write(6,*) "Fupper CnMax returned error ",Istat,
     1      " for Method ",Method0," experiment 2"
             EndIf
             If(CL.lt. 0.99999 .and. mu2 .lt. 99.) then
                p2=CMaxCL(CL,0.,mu2)
             Else
                p2=1.
             EndIf
           EndIf
C If either p1 or p2 are .ge. 1, act as if CombConf=1, so the fit will have
C to use a lower sigma to get CombConf-CL0 = 0.
           If(p1 .lt. 1. .and. p2 .lt. 1.) Then
             Fupper=CombConf(abs(Method0),p1,p2,mu1,mu2,IFlag0)-CL0
           Else
             Fupper=1.-CL0
           EndIf
           If(IFlag0.gt.3) Then
              Write(6,*) "Fupper CombConf finds error",IFlag0
           EndIf
        EndIf
      EndIf
      Return
      End
