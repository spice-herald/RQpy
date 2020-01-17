      Real Function CombConf(Method,p1,p2,mua,mub,ICode)
C Gives the confidence level by which the cross section is rejected as too
C high when it is rejected by two measurements with confidence levels p1 and
C p2.  The two measurements have total expected number of events mu1 and mu2.
C The Method is as numbered in CDMS_Edelweiss.pdf.
C ICode is as in C0Int except it's 5 if XC0 has a problem.
      Implicit None
      Integer Method,ICode1,ICode2,ICode
      Real p1,p2,mu1,mu2,z,x1,x2,mua,mub,x,XC0,C0,C0Int
C      Real tmp1,tmp2,tmp3,tmp4
C
      ICode=5
      CombConf=0.
      mu1=max(mua,mub)
      mu2=min(mua,mub)
      If(Method.lt.3 .or. Method.gt.6) Then
         Write(6,*) "This Method is not implemented"
      ElseIf(Method.eq.3) Then
         x1=XC0(p1,mua,ICode1)
         x2=XC0(p2,mub,ICode2)
         If(ICode1.ne.0 .or. ICode2.ne.0) Then
            Write(6,*) "CombConf trouble computing XC0.",p1,p2,mua,
     1        mub,ICode1,ICode2
            CombConf=0.
            ICode=5
            Return
         Else
            ICode=0
         EndIf
         z=x1+x2
         If(z.le.0.) Then
            CombConf=0.
         ElseIf(z.le.mu2) Then
            CombConf=C0Int(0.,z,mu1,mu2,z,ICode)
         ElseIf(z.le.mu1) Then
            CombConf=C0(z-mu2,mu1) + C0Int(z-mu2,z,mu1,mu2,z,ICode)
         ElseIf( z .lt. (mu1+mu2)*.99999 ) Then
            CombConf=C0(z-mu2,mu1) + exp(-mu1)*C0(z-mu1,mu2)
     2       + C0Int(z-mu2,mu1,mu1,mu2,z,ICode)
         ElseIf( z .lt. 1.00001*(mu1+mu2)) Then
C Allow for some roundoff error with no events in either experiment
            CombConf=1.-exp(-(mu1+mu2))
         Else
            CombConf=1.
         EndIf
      ElseIf(Method.eq.4) Then
         ICode=0
         z=max(p1,p2)
         If(z.le.0.) Then
            CombConf=0.
         ElseIf(z.lt.1.-exp(-mu2)) Then
            CombConf=z*z
         ElseIf(z.lt.1.-exp(-mu1)) Then
            CombConf=z
         Else
            CombConf=1.
         EndIf
      ElseIf(Method.eq.5) Then
         ICode=0
         z=min(p1,p2)
         If(z.le.0) Then
            CombConf=0.
         ElseIf(z.lt.1.00001-exp(-mu2)) Then
            CombConf=z*(2.-z)
         Else
            CombConf=1.
         EndIf
      ElseIf(Method.eq.6) Then
         ICode=0
         z=p1*p2
         x=(1.-exp(-mu1))*(1.-exp(-mu2))
         If(z.le.0.) Then
            CombConf=0.
         ElseIf(z.lt.x+0.00001) Then
            CombConf=z*( (1.-exp(-(mu1+mu2)))/x + log(x/z) )
         Else
            CombConf=1.
         EndIf
      Else
         Write(6,*) "You should never get here"
      EndIf
      Return
      End

