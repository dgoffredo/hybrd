#include <cstdlib>
#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <ctime>

#include <algorithm>
#include <numeric>
#include <limits>
#include <vector>
#include <cassert>

#include "minpack.hpp"

//****************************************************************************80

void chkder ( int m, int n, double x[], double fvec[], double fjac[],
  int ldfjac, double xp[], double fvecp[], int mode, double err[] )

//****************************************************************************80
//
//  Purpose:
//
//    CHKDER checks the gradients of M functions in N variables.
//
//  Discussion:
//
//    This subroutine checks the gradients of M nonlinear functions
//    in N variables, evaluated at a point x, for consistency with
//    the functions themselves. the user must call chkder twice,
//    first with mode = 1 and then with mode = 2.
//
//    mode = 1: on input, x must contain the point of evaluation.
//    on output, xp is set to a neighboring point.
//
//    mode = 2. on input, fvec must contain the functions and the
//    rows of fjac must contain the gradients
//    of the respective functions each evaluated
//    at x, and fvecp must contain the functions
//    evaluated at xp.
//    on output, err contains measures of correctness of
//    the respective gradients.
//
//    the subroutine does not perform reliably if cancellation or
//    rounding errors cause a severe loss of significance in the
//    evaluation of a function. therefore, none of the components
//    of x should be unusually small (in particular, zero) or any
//    other value which may cause loss of significance.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       m is a positive integer input variable set to the number
//         of functions.
//
//       n is a positive integer input variable set to the number
//         of variables.
//
//       x is an input array of length n.
//
//       fvec is an array of length m. on input when mode = 2,
//         fvec must contain the functions evaluated at x.
//
//       fjac is an m by n array. on input when mode = 2,
//         the rows of fjac must contain the gradients of
//         the respective functions evaluated at x.
//
//       ldfjac is a positive integer input parameter not less than m
//         which specifies the leading dimension of the array fjac.
//
//       xp is an array of length n. on output when mode = 1,
//         xp is set to a neighboring point of x.
//
//       fvecp is an array of length m. on input when mode = 2,
//         fvecp must contain the functions evaluated at xp.
//
//       mode is an integer input variable set to 1 on the first call
//         and 2 on the second. other values of mode are equivalent
//         to mode = 1.
//
//       err is an array of length m. on output when mode = 2,
//         err contains measures of correctness of the respective
//         gradients. if there is no severe loss of significance,
//         then if err(i) is 1.0 the i-th gradient is correct,
//         while if err(i) is 0.0 the i-th gradient is incorrect.
//         for values of err between 0.0 and 1.0, the categorization
//         is less certain. in general, a value of err(i) greater
//         than 0.5 indicates that the i-th gradient is probably
//         correct, while a value of err(i) less than 0.5 indicates
//         that the i-th gradient is probably incorrect.
//
{
  using std::sqrt;
  using std::log10;

  double eps;
  double epsf;
  double epslog;
  double epsmch;
  double factor = 100.0;
  int i;
  int j;
  double temp;
//
//  EPSMCH is the machine precision.
//
  epsmch = std::numeric_limits<double>::epsilon();
//
  eps = sqrt ( epsmch );
//
//  MODE = 1.
//
  if ( mode == 1 )
  {
    for ( j = 0; j < n; j++ )
    {
      if ( x[j] == 0.0 )
      {
        temp = eps;
      }
      else
      {
        temp = eps * std::fabs ( x[j] );
      }
      xp[j] = x[j] + temp;
    }
  }
//
//  MODE = 2.
//
  else
  {
    epsf = factor * epsmch;
    epslog = log10 ( eps );
    for ( i = 0; i < m; i++ )
    {
      err[i] = 0.0;
    }

    for ( j = 0; j < n; j++ )
    {
      if ( x[j] == 0.0 )
      {
        temp = 1.0;
      }
      else
      {
        temp = std::fabs ( x[j] );
      }
      for ( i = 0; i < m; i++ )
      {
        err[i] = err[i] + temp * fjac[i+j*ldfjac];
      }
    }

    for ( i = 0; i < m; i++ )
    {
      temp = 1.0;
      if ( fvec[i] != 0.0 &&
           fvecp[i] != 0.0 &&
           epsf * std::fabs ( fvec[i] ) <= std::fabs ( fvecp[i] - fvec[i] ) )
      {
        temp = eps * std::fabs ( ( fvecp[i] - fvec[i] ) / eps - err[i] )
          / ( std::fabs ( fvec[i] ) + std::fabs ( fvecp[i] ) );

        if ( temp <= epsmch )
        {
          err[i] = 1.0;
        }
        else if ( temp < eps )
        {
          err[i] = ( log10 ( temp ) - epslog ) / epslog;
        }
        else
        {
          err[i] = 0.0;
        }
      }
    }
  }
}
//****************************************************************************80

void dogleg ( int n, double r[], int lr, double diag[], double qtb[],
  double delta, double x[], double wa1[], double wa2[] )

//****************************************************************************80
//
//  Purpose:
//
//    DOGLEG combines Gauss-Newton and gradient for a minimizing step.
//
//  Discussion:
//
//    Given an m by n matrix a, an n by n nonsingular diagonal
//    matrix d, an m-vector b, and a positive number delta, the
//    problem is to determine the convex combination x of the
//    gauss-newton and scaled gradient directions that minimizes
//    (a*x - b) in the least squares sense, subject to the
//    restriction that the euclidean norm of d*x be at most delta.
//
//    This subroutine completes the solution of the problem
//    if it is provided with the necessary information from the
//    qr factorization of a. that is, if a = q*r, where q has
//    orthogonal columns and r is an upper triangular matrix,
//    then dogleg expects the full upper triangle of r and
//    the first n components of (q transpose)*b.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       n is a positive integer input variable set to the order of r.
//
//       r is an input array of length lr which must contain the upper
//         triangular matrix r stored by rows.
//
//       lr is a positive integer input variable not less than
//         (n*(n+1))/2.
//
//       diag is an input array of length n which must contain the
//         diagonal elements of the matrix d.
//
//       qtb is an input array of length n which must contain the first
//         n elements of the vector (q transpose)*b.
//
//       delta is a positive input variable which specifies an upper
//         bound on the euclidean norm of d*x.
//
//       x is an output array of length n which contains the desired
//         convex combination of the gauss-newton direction and the
//         scaled gradient direction.
//
//       wa1 and wa2 are work arrays of length n.
//
{
  using std::pow;
  using std::sqrt;

  double alpha;
  double bnorm;
  double epsmch;
  double gnorm;
  int i;
  int j;
  int jj;
  int jp1;
  int k;
  int l;
  double qnorm;
  double sgnorm;
  double sum;
  double temp;
//
//  EPSMCH is the machine precision.
//
  epsmch = std::numeric_limits<double>::epsilon();
//
//  Calculate the Gauss-Newton direction.
//
  jj = ( n * ( n + 1 ) ) / 2 + 1;

  for ( k = 1; k <= n; k++ )
  {
    j = n - k + 1;
    jp1 = j + 1;
    jj -=  k;
    l = jj + 1;
    sum = 0.0;
    for ( i = jp1; i <= n; i++ )
    {
      sum +=  r[l-1] * x[i-1];
      l +=  1;
    }
    temp = r[jj-1];
    if ( temp == 0.0 )
    {
      l = j;
      for ( i = 1; i <= j; i++ )
      {
        temp = std::max ( temp, std::fabs ( r[l-1] ) );
        l +=  n - i;
      }
      temp = epsmch * temp;
      if ( temp == 0.0 )
      {
        temp = epsmch;
      }
    }
    x[j-1] = ( qtb[j-1] - sum ) / temp;
  }
//
//  Test whether the Gauss-Newton direction is acceptable.
//
  for ( j = 0; j < n; j++ )
  {
    wa1[j] = 0.0;
    wa2[j] = diag[j] * x[j];
  }
  qnorm = enorm ( n, wa2 );

  if ( qnorm <= delta )
  {
    return;
  }
//
//  The Gauss-Newton direction is not acceptable.
//  Calculate the scaled gradient direction.
//
  l = 0;
  for ( j = 0; j < n; j++ )
  {
    temp = qtb[j];
    for ( i = j; i < n; i++ )
    {
      wa1[i-1] = wa1[i-1] + r[l-1] * temp;
      l +=  1;
    }
    wa1[j] = wa1[j] / diag[j];
  }
//
//  Calculate the norm of the scaled gradient and test for
//  the special case in which the scaled gradient is zero.
//
  gnorm = enorm ( n, wa1 );
  sgnorm = 0.0;
  alpha = delta / qnorm;
//
//  Calculate the point along the scaled gradient
//  at which the quadratic is minimized.
//
  if ( gnorm != 0.0 )
  {
    for ( j = 0; j < n; j++ )
    {
      wa1[j] = ( wa1[j] / gnorm ) / diag[j];
    }
    l = 0;
    for ( j = 0; j < n; j++ )
    {
      sum = 0.0;
      for ( i = j; i < n; i++ )
      {
        sum +=  r[l] * wa1[i];
        l +=  1;
      }
      wa2[j] = sum;
    }
    temp = enorm ( n, wa2 );
    sgnorm = ( gnorm / temp ) / temp;
    alpha = 0.0;
//
//  If the scaled gradient direction is not acceptable,
//  calculate the point along the dogleg at which the quadratic is minimized.
//
    if ( sgnorm < delta)
    {
      bnorm = enorm ( n, qtb );
      temp = ( bnorm / gnorm ) * ( bnorm / qnorm ) * ( sgnorm / delta );
      temp -=  ( delta / qnorm ) * ( sgnorm / delta ) * ( sgnorm / delta )
        - sqrt ( pow ( temp - ( delta / qnorm ), 2 )
        + ( 1.0 - ( delta / qnorm ) * ( delta / qnorm ) )
        * ( 1.0 - ( sgnorm / delta ) * ( sgnorm / delta ) ) );
      alpha = ( ( delta / qnorm )
        * ( 1.0 - ( sgnorm / delta ) * ( sgnorm / delta ) ) ) / temp;
    }
  }
//
//  Form appropriate convex combination of the Gauss-Newton
//  direction and the scaled gradient direction.
//
  temp = ( 1.0 - alpha ) * std::min ( sgnorm, delta );
  for ( j = 0; j < n; j++ )
  {
    x[j] = temp * wa1[j] + alpha * x[j];
  }
}
//****************************************************************************80
namespace {
    struct AddSquare {
        template <typename T> 
        T operator () (T sum, T term) { return sum + term * term; }
    };
} 

double enorm ( int n, double x[] )

//****************************************************************************80
//
//  Purpose:
//
//    ENORM returns the Euclidean norm of a vector.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010 // John Burkhardt
//    25 August 2013 // David Goffredo    
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of entries in A.
//
//    Input, double X[N], the vector whose norm is desired.
//
//    Output, double ENORM, the norm of X.
//
{
  return std::sqrt(std::accumulate(x, x + n, 0.0, AddSquare()));
  /*
  double value = 0.0;

  for (int i = 0; i < n; i++ )
  {
    value += x[i] * x[i];
  }
  return std::sqrt ( value );
  */
}
//****************************************************************************80

template <typename Func>
void fdjac1 ( Func fcn,
  int n, double x[], double fvec[], double fjac[], int ldfjac, int *iflag,
  int ml, int mu, double epsfcn, double wa1[], double wa2[] )

//****************************************************************************80
//
//  Purpose:
//
//    FDJAC1 estimates an N by N Jacobian matrix using forward differences.
//
//  Discussion:
//
//    This subroutine computes a forward-difference approximation
//    to the n by n jacobian matrix associated with a specified
//    problem of n functions in n variables. if the jacobian has
//    a banded form, then function evaluations are saved by only
//    approximating the nonzero terms.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       fcn is the name of the user-supplied subroutine which
//         calculates the functions. fcn must be declared
//         in an external statement in the user calling
//         program, and should be written as follows.
//
//         subroutine fcn(n,x,fvec,iflag)
//         integer n,iflag
//         double precision x(n),fvec(n)
//         ----------
//         calculate the functions at x and
//         return this vector in fvec.
//         ----------
//         return
//         end
//
//         the value of iflag should not be changed by fcn unless
//         the user wants to terminate execution of fdjac1.
//         in this case set iflag to a negative integer.
//
//       n is a positive integer input variable set to the number
//         of functions and variables.
//
//       x is an input array of length n.
//
//       fvec is an input array of length n which must contain the
//         functions evaluated at x.
//
//       fjac is an output n by n array which contains the
//         approximation to the jacobian matrix evaluated at x.
//
//       ldfjac is a positive integer input variable not less than n
//         which specifies the leading dimension of the array fjac.
//
//       iflag is an integer variable which can be used to terminate
//         the execution of fdjac1. see description of fcn.
//
//       ml is a nonnegative integer input variable which specifies
//         the number of subdiagonals within the band of the
//         jacobian matrix. if the jacobian is not banded, set
//         ml to at least n - 1.
//
//       epsfcn is an input variable used in determining a suitable
//         step length for the forward-difference approximation. this
//         approximation assumes that the relative errors in the
//         functions are of the order of epsfcn. if epsfcn is less
//         than the machine precision, it is assumed that the relative
//         errors in the functions are of the order of the machine
//         precision.
//
//       mu is a nonnegative integer input variable which specifies
//         the number of superdiagonals within the band of the
//         jacobian matrix. if the jacobian is not banded, set
//         mu to at least n - 1.
//
//       wa1 and wa2 are work arrays of length n. if ml + mu + 1 is at
//         least n, then the jacobian is considered dense, and wa2 is
//         not referenced.
{
  double eps;
  double epsmch;
  double h;
  int i;
  int j;
  int k;
  int msum;
  double temp;
//
//  EPSMCH is the machine precision.
//
  epsmch = std::numeric_limits<double>::epsilon();

  eps = sqrt ( std::max ( epsfcn, epsmch ) );
  msum = ml + mu + 1;
//
//  Computation of dense approximate jacobian.
//
  if ( n <= msum )
  {
    for ( j = 0; j < n; j++ )
    {
      temp = x[j];
      h = eps * std::fabs ( temp );
      if ( h == 0.0 )
      {
        h = eps;
      }
      x[j] = temp + h;
      fcn ( n, x, wa1, iflag );
      if ( *iflag < 0 )
      {
        break;
      }
      x[j] = temp;
      for ( i = 0; i < n; i++ )
      {
        fjac[i+j*ldfjac] = ( wa1[i] - fvec[i] ) / h;
      }
    }
  }
//
//  Computation of banded approximate jacobian.
//
  else
  {
    for ( k = 0; k < msum; k++ )
    {
      for ( j = k; j < n; j +=  msum )
      {
        wa2[j] = x[j];
        h = eps * std::fabs ( wa2[j] );
        if ( h == 0.0 )
        {
          h = eps;
        }
        x[j] = wa2[j] + h;
      }
      fcn ( n, x, wa1, iflag );
      if ( *iflag < 0 )
      {
        break;
      }
      for ( j = k; j < n; j +=  msum )
      {
        x[j] = wa2[j];
        h = eps * std::fabs ( wa2[j] );
        if ( h == 0.0 )
        {
          h = eps;
        }
        for ( i = 0; i < n; i++ )
        {
          if ( j - mu <= i && i <= j + ml )
          {
            fjac[i+j*ldfjac] = ( wa1[i] - fvec[i] ) / h;
          }
          else
          {
            fjac[i+j*ldfjac] = 0.0;
          }
        }
      }
    }
  }
}
//****************************************************************************80

template <typename Func>
void fdjac2 ( Func fcn,
  int m, int n, double x[], double fvec[], double fjac[], int ldfjac,
  int *iflag, double epsfcn, double wa[] )

//****************************************************************************80
//
//  Purpose:
//
//    FDJAC2 estimates an M by N Jacobian matrix using forward differences.
//
//  Discussion:
//
//    This subroutine computes a forward-difference approximation
//    to the m by n jacobian matrix associated with a specified
//    problem of m functions in n variables.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       fcn is the name of the user-supplied subroutine which
//         calculates the functions. fcn must be declared
//         in an external statement in the user calling
//         program, and should be written as follows.
//
//         subroutine fcn(m,n,x,fvec,iflag)
//         integer m,n,iflag
//         double precision x(n),fvec(m)
//         ----------
//         calculate the functions at x and
//         return this vector in fvec.
//         ----------
//         return
//         end
//
//         the value of iflag should not be changed by fcn unless
//         the user wants to terminate execution of fdjac2.
//         in this case set iflag to a negative integer.
//
//       m is a positive integer input variable set to the number
//         of functions.
//
//       n is a positive integer input variable set to the number
//         of variables. n must not exceed m.
//
//       x is an input array of length n.
//
//       fvec is an input array of length m which must contain the
//         functions evaluated at x.
//
//       fjac is an output m by n array which contains the
//         approximation to the jacobian matrix evaluated at x.
//
//       ldfjac is a positive integer input variable not less than m
//         which specifies the leading dimension of the array fjac.
//
//       iflag is an integer variable which can be used to terminate
//         the execution of fdjac2. see description of fcn.
//
//       epsfcn is an input variable used in determining a suitable
//         step length for the forward-difference approximation. this
//         approximation assumes that the relative errors in the
//         functions are of the order of epsfcn. if epsfcn is less
//         than the machine precision, it is assumed that the relative
//         errors in the functions are of the order of the machine
//         precision.
//
//       wa is a work array of length m.
//
{
  double eps;
  double epsmch;
  double h;
  int i;
  int j;
  double temp;
//
//  EPSMCH is the machine precision.
//
  epsmch = std::numeric_limits<double>::epsilon();
  eps = sqrt ( std::max ( epsfcn, epsmch ) );

  for ( j = 0; j < n; j++ )
  {
    temp = x[j];
    if ( temp == 0.0 )
    {
      h = eps;
    }
    else
    {
      h = eps * std::fabs ( temp );
    }
    x[j] = temp + h;
    fcn ( m, n, x, wa, iflag );
    if ( *iflag < 0 )
    {
      break;
    }
    x[j] = temp;
    for ( i = 0; i < m; i++ )
    {
      fjac[i+j*ldfjac] = ( wa[i] - fvec[i] ) / h;
    }
  }
}
//****************************************************************************80

template <typename Func>
int hybrd ( Func fcn,
  int n, double x[],
  double fvec[], double xtol, int maxfev, int ml, int mu, double epsfcn,
  double diag[], int mode, double factor, int nprint, int nfev,
  double fjac[], int ldfjac, double r[], int lr, double qtf[], double wa1[],
  double wa2[], double wa3[], double wa4[] )
  
//****************************************************************************80
//
//  Purpose:
//
//    HYBRD finds a zero of a system of N nonlinear equations.
//
//  Discussion:
//
//    The purpose of hybrd is to find a zero of a system of
//    n nonlinear functions in n variables by a modification
//    of the powell hybrid method. the user must provide a
//    subroutine which calculates the functions. the jacobian is
//    then calculated by a forward-difference approximation.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       fcn is the name of the user-supplied subroutine which
//         calculates the functions. fcn must be declared
//         in an external statement in the user calling
//         program, and should be written as follows.
//
//         subroutine fcn(n,x,fvec,iflag)
//         integer n,iflag
//         double precision x(n),fvec(n)
//         ----------
//         calculate the functions at x and
//         return this vector in fvec.
//         ---------
//         return
//         end
//
//         the value of iflag should not be changed by fcn unless
//         the user wants to terminate execution of hybrd.
//         in this case set iflag to a negative integer.
//
//       n is a positive integer input variable set to the number
//         of functions and variables.
//
//       x is an array of length n. on input x must contain
//         an initial estimate of the solution vector. on output x
//         contains the final estimate of the solution vector.
//
//       fvec is an output array of length n which contains
//         the functions evaluated at the output x.
//
//       xtol is a nonnegative input variable. termination
//         occurs when the relative error between two consecutive
//         iterates is at most xtol.
//
//       maxfev is a positive integer input variable. termination
//         occurs when the number of calls to fcn is at least maxfev
//         by the end of an iteration.
//
//       ml is a nonnegative integer input variable which specifies
//         the number of subdiagonals within the band of the
//         jacobian matrix. if the jacobian is not banded, set
//         ml to at least n - 1.
//
//       mu is a nonnegative integer input variable which specifies
//         the number of superdiagonals within the band of the
//         jacobian matrix. if the jacobian is not banded, set
//         mu to at least n - 1.
//
//       epsfcn is an input variable used in determining a suitable
//         step length for the forward-difference approximation. this
//         approximation assumes that the relative errors in the
//         functions are of the order of epsfcn. if epsfcn is less
//         than the machine precision, it is assumed that the relative
//         errors in the functions are of the order of the machine
//         precision.
//0
//       diag is an array of length n. if mode = 1 (see
//         below), diag is internally set. if mode = 2, diag
//         must contain positive entries that serve as
//         multiplicative scale factors for the variables.
//
//       mode is an integer input variable. if mode = 1, the
//         variables will be scaled internally. if mode = 2,
//         the scaling is specified by the input diag. other
//         values of mode are equivalent to mode = 1.
//
//       factor is a positive input variable used in determining the
//         initial step bound. this bound is set to the product of
//         factor and the euclidean norm of diag*x if nonzero, or else
//         to factor itself. in most cases factor should lie in the
//         interval (.1,100.). 100. is a generally recommended value.
//
//       nprint is an integer input variable that enables controlled
//         printing of iterates if it is positive. in this case,
//         fcn is called with iflag = 0 at the beginning of the first
//         iteration and every nprint iterations thereafter and
//         immediately prior to return, with x and fvec available
//         for printing. if nprint is not positive, no special calls
//         of fcn with iflag = 0 are made.
//
//       info is an integer output variable. if the user has
//         terminated execution, info is set to the (negative)
//         value of iflag. see description of fcn. otherwise,
//         info is set as follows.
//
//         info = 0   improper input parameters.
//
//         info = 1   relative error between two consecutive iterates
//                    is at most xtol.
//
//         info = 2   number of calls to fcn has reached or exceeded
//                    maxfev.
//
//         info = 3   xtol is too small. no further improvement in
//                    the approximate solution x is possible.
//
//         info = 4   iteration is not making good progress, as
//                    measured by the improvement from the last
//                    five jacobian evaluations.
//
//         info = 5   iteration is not making good progress, as
//                    measured by the improvement from the last
//                    ten iterations.
//
//       nfev is an integer output variable set to the number of
//         calls to fcn.
//
//       fjac is an output n by n array which contains the
//         orthogonal matrix q produced by the qr factorization
//         of the final approximate jacobian.
//
//       ldfjac is a positive integer input variable not less than n
//         which specifies the leading dimension of the array fjac.
//
//       r is an output array of length lr which contains the
//         upper triangular matrix produced by the qr factorization
//         of the final approximate jacobian, stored rowwise.
//
//       lr is a positive integer input variable not less than
//         (n*(n+1))/2.
//
//       qtf is an output array of length n which contains
//         the vector (q transpose)*fvec.
//
//       wa1, wa2, wa3, and wa4 are work arrays of length n.
//
{
  double actred;
  double delta;
  double epsmch;
  double fnorm;
  double fnorm1;
  int i;
  int iflag;
  int info;
  int iter;
  int iwa[1];
  int j;
  bool jeval;
  int jm1;
  int l;
  int msum;
  int ncfail;
  int ncsuc;
  int nslow1;
  int nslow2;
  // double 0.001 = 0.001;
  // double 0.0001 = 0.0001;
  // double 0.1 = 0.1;
  // double 0.5 = 0.5;
  double pnorm;
  double prered;
  double ratio;
  bool sing;
  double sum;
  double temp;
  double xnorm;
//
//  Certain loops in this function were kept closer to their original FORTRAN77
//  format, to avoid confusing issues with the array index L.  These loops are
//  marked "DO NOT ADJUST", although they certainly could be adjusted (carefully)
//  once the initial translated code is tested.
//

//
//  EPSMCH is the machine precision.
//
  epsmch = std::numeric_limits<double>::epsilon();

  info = 0;
  iflag = 0;
  nfev = 0;
//
//  Check the input parameters.
//
  if ( n <= 0 )
  {
    info = 0;
    return info;
  }
  if ( xtol < 0.0 )
  {
    info = 0;
    return info;
  }
  if ( maxfev <= 0 )
  {
    info = 0;
    return info;
  }
  if ( ml < 0 )
  {
    info = 0;
    return info;
  }
  if ( mu < 0 )
  {
    info = 0;
    return info;
  }
  if ( factor <= 0.0 )
  {
    info = 0;
    return info;
  }
  if ( ldfjac < n )
  {
    info = 0;
    return info;
  }
  if ( lr < ( n * ( n + 1 ) ) / 2 )
  {
    info = 0;
    return info;
  }
  if ( mode == 2 )
  {
    for ( j = 0; j < n; j++ )
    {
      if ( diag[j] <= 0.0 )
      {
        info = 0;
        return info;
      }
    }
  }
//
//  Evaluate the function at the starting point and calculate its norm.
//
  iflag = 1;
  fcn ( n, x, fvec, &iflag );
  nfev = 1;
  if ( iflag < 0 )
  {
    info = iflag;
    return info;
  }

  fnorm = enorm ( n, fvec );
//
//  Determine the number of calls to FCN needed to compute the jacobian matrix.
//
  msum = std::min ( ml + mu + 1, n );
//
//  Initialize iteration counter and monitors.
//
  iter = 1;
  ncsuc = 0;
  ncfail = 0;
  nslow1 = 0;
  nslow2 = 0;
//
//  Beginning of the outer loop.
//
  for ( ; ; )
  {
    jeval = true;
//
//  Calculate the jacobian matrix.
//
    iflag = 2;
    fdjac1 ( fcn, n, x, fvec, fjac, ldfjac, &iflag, ml, mu, epsfcn, wa1, wa2 );

    nfev +=  msum;
    if ( iflag < 0 )
    {
      info = iflag;
      return info;
    }
//
//  Compute the QR factorization of the jacobian.
//
    qrfac ( n, n, fjac, ldfjac, false, iwa, 1, wa1, wa2, wa3 );
//
//  On the first iteration and if MODE is 1, scale according
//  to the norms of the columns of the initial jacobian.
//
    if ( iter == 1 )
    {
      if ( mode == 1 )
      {
        for ( j = 0; j < n; j++ )
        {
          if ( wa2[j] != 0.0 )
          {
            diag[j] = wa2[j];
          }
          else
          {
            diag[j] = 1.0;
          }
        }
      }
//
//  On the first iteration, calculate the norm of the scaled X
//  and initialize the step bound DELTA.
//
      for ( j = 0; j < n; j++ )
      {
        wa3[j] = diag[j] * x[j];
      }
      xnorm = enorm ( n, wa3 );

      if ( xnorm == 0.0 )
      {
        delta = factor;
      }
      else
      {
        delta = factor * xnorm;
      }
    }
//
//  Form Q' * FVEC and store in QTF.
//
    for ( i = 0; i < n; i++ )
    {
      qtf[i] = fvec[i];
    }
    for ( j = 0; j < n; j++ )
    {
      if ( fjac[j+j*ldfjac] != 0.0 )
      {
        sum = 0.0;
        for ( i = j; i < n; i++ )
        {
          sum +=  fjac[i+j*ldfjac] * qtf[i];
        }
        temp = - sum / fjac[j+j*ldfjac];
        for ( i = j; i < n; i++ )
        {
          qtf[i] = qtf[i] + fjac[i+j*ldfjac] * temp;
        }
      }
    }
//
//  Copy the triangular factor of the QR factorization into R.
//
//  DO NOT ADJUST THIS LOOP, BECAUSE OF L.
//
    sing = false;
    for ( j = 1; j <= n; j++ )
    {
      l = j;
      for ( i = 1; i <= j - 1; i++ )
      {
        r[l-1] = fjac[(i-1)+(j-1)*ldfjac];
        l +=  n - i;
      }
      r[l-1] = wa1[j-1];
      if ( wa1[j-1] == 0.0 )
      {
        sing = true;
      }
    }
//
//  Accumulate the orthogonal factor in FJAC.
//
    qform ( n, n, fjac, ldfjac, wa1 );
//
//  Rescale if necessary.
//
    if ( mode == 1 )
    {
      for ( j = 0; j < n; j++ )
      {
        diag[j] = std::max ( diag[j], wa2[j] );
      }
    }
//
//  Beginning of the inner loop.
//
    for ( ; ; )
    {
//
//  If requested, call FCN to enable printing of iterates.
//
      if ( 0 < nprint )
      {
        if ( ( iter - 1 ) % nprint == 0 )
        {
          iflag = 0;
          fcn ( n, x, fvec, &iflag );
          if ( iflag < 0 )
          {
            info = iflag;
            return info;
          }
        }
      }
//
//  Determine the direction P.
//
      dogleg ( n, r, lr, diag, qtf, delta, wa1, wa2, wa3 );
//
//  Store the direction P and X + P.  Calculate the norm of P.
//
      for ( j = 0; j < n; j++ )
      {
        double wa1j = wa1[j] = - wa1[j];
        wa2[j] = x[j] + wa1j;
        wa3[j] = diag[j] * wa1j;
      }
      pnorm = enorm ( n, wa3 );
//
//  On the first iteration, adjust the initial step bound.
//
      if ( iter == 1 )
      {
        delta = std::min ( delta, pnorm );
      }
//
//  Evaluate the function at X + P and calculate its norm.
//
      iflag = 1;
      fcn ( n, wa2, wa4, &iflag );
      nfev +=  1;
      if ( iflag < 0 )
      {
        info = iflag;
        return info;
      }
      fnorm1 = enorm ( n, wa4 );
//
//  Compute the scaled actual reduction.
//
      if ( fnorm1 < fnorm )
      {
        actred = 1.0 - ( fnorm1 / fnorm ) * ( fnorm1 / fnorm );
      }
      else
      {
        actred = - 1.0;
      }
//
//  Compute the scaled predicted reduction.
//
//  DO NOT ADJUST THIS LOOP, BECAUSE OF L.
//
      l = 1;
      for ( i = 1; i <= n; i++ )
      {
        sum = 0.0;
        for ( j = i; j <= n; j++ )
        {
          sum +=  r[l-1] * wa1[j-1];
          l +=  1;
        }
        wa3[i-1] = qtf[i-1] + sum;
      }
      temp = enorm ( n, wa3 );

      if ( temp < fnorm )
      {
        prered = 1.0 - ( temp / fnorm ) * ( temp / fnorm );
      }
      else
      {
        prered = 0.0;
      }
//
//  Compute the ratio of the actual to the predicted reduction.
//
      if ( 0.0 < prered )
      {
        ratio = actred / prered;
      }
      else
      {
        ratio = 0.0;
      }
//
//  Update the step bound.
//
      if ( ratio < 0.1 )
      {
        ncsuc = 0;
        ncfail +=  1;
        delta = 0.5 * delta;
      }
      else
      {
        ncfail = 0;
        ncsuc +=  1;
        if ( 0.5 <= ratio || 1 < ncsuc )
        {
          delta = std::max ( delta, pnorm / 0.5 );
        }
        if ( std::fabs ( ratio - 1.0 ) <= 0.1 )
        {
          delta = pnorm / 0.5;
        }
      }
//
//  On successful iteration, update X, FVEC, and their norms.
//
      if ( 0.0001 <= ratio )
      {
        for ( j = 0; j < n; j++ )
        {
          x[j] = wa2[j];
          wa2[j] = diag[j] * x[j];
          fvec[j] = wa4[j];
        }
        xnorm = enorm ( n, wa2 );
        fnorm = fnorm1;
        iter +=  1;
      }
//
//  Determine the progress of the iteration.
//
      nslow1 +=  1;
      if ( 0.001 <= actred )
      {
        nslow1 = 0;
      }
      if ( jeval )
      {
        nslow2 +=  1;
      }
      if ( 0.1 <= actred )
      {
        nslow2 = 0;
      }
//
//  Test for convergence.
//
      if ( delta <= xtol * xnorm || fnorm == 0.0 )
      {
        info = 1;
        return info;
      }
//
//  Tests for termination and std::stringent tolerances.
//
      if ( maxfev <= nfev )
      {
        info = 2;
        return info;
      }
      if ( 0.1 * std::max ( 0.1 * delta, pnorm ) <= epsmch * xnorm )
      {
        info = 3;
        return info;
      }
      if ( nslow2 == 5 )
      {
        info = 4;
        return info;
      }
      if ( nslow1 == 10 )
      {
        info = 5;
        return info;
      }
//
//  Criterion for recalculating jacobian approximation by forward differences.
//
      if ( ncfail == 2 )
      {
        break;
      }
//
//  Calculate the rank one modification to the jacobian
//  and update QTF if necessary.
//
      for ( j = 0; j < n; j++ )
      {
        sum = 0.0;
        for ( i = 0; i < n; i++ )
        {
          sum += fjac[i+j*ldfjac] * wa4[i];
        }
        wa2[j] = ( sum - wa3[j] ) / pnorm;
        wa1[j] = diag[j] * ( ( diag[j] * wa1[j] ) / pnorm );
        if ( 0.0001 <= ratio )
        {
          qtf[j] = sum;
        }
      }
//
//  Compute the QR factorization of the updated jacobian.
//
      sing = r1updt ( n, n, r, lr, wa1, wa2, wa3 );
      r1mpyq ( n, n, fjac, ldfjac, wa2, wa3 );
      r1mpyq ( 1, n, qtf, 1, wa2, wa3 );

      jeval = false;
    }
//
//  End of the inner loop.
//
  }
//
//  End of the outer loop.
//
}
//****************************************************************************80

template <typename Func>
int hybrd1 ( Func fcn, int n,
  double x[], double fvec[], double tol, double wa[], int lwa )
//                                                    lwa >= (n*(3*n+13))/2.

//****************************************************************************80
//
//  Purpose:
//
//    HYBRD1 is a simplified interface to HYBRD.
//
//  Discussion:
//
//    The purpose of HYBRD1 is to find a zero of a system of
//    N nonlinear functions in N variables by a modification
//    of the Powell hybrid method.  This is done by using the
//    more general nonlinear equation solver HYBRD.  The user
//    must provide a subroutine which calculates the functions.
//    The jacobian is then calculated by a forward-difference
//    approximation.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       fcn is the name of the user-supplied subroutine which
//         calculates the functions. fcn must be declared
//         in an external statement in the user calling
//         program, and should be written as follows.
//
//         subroutine fcn(n,x,fvec,iflag)
//         integer n,iflag
//         double precision x(n),fvec(n)
//         ----------
//         calculate the functions at x and
//         return this vector in fvec.
//         ---------
//         return
//         end
//
//         the value of iflag should not be changed by fcn unless
//         the user wants to terminate execution of hybrd1.
//         in this case set iflag to a negative integer.
//
//       n is a positive integer input variable set to the number
//         of functions and variables.
//
//       x is an array of length n. on input x must contain
//         an initial estimate of the solution vector. on output x
//         contains the final estimate of the solution vector.
//
//       fvec is an output array of length n which contains
//         the functions evaluated at the output x.
//
//       tol is a nonnegative input variable. termination occurs
//         when the algorithm estimates that the relative error
//         between x and the solution is at most tol.
//
//       info is an integer output variable. if the user has
//         terminated execution, info is set to the (negative)
//         value of iflag. see description of fcn. otherwise,
//         info is set as follows.
//
//         info = 0   improper input parameters.
//
//         info = 1   algorithm estimates that the relative error
//                    between x and the solution is at most tol.
//
//         info = 2   number of calls to fcn has reached or exceeded
//                    200*(n+1).
//
//         info = 3   tol is too small. no further improvement in
//                    the approximate solution x is possible.
//
//         info = 4   iteration is not making good progress.
//
//       wa is a work array of length lwa.
//
//       lwa is a positive integer input variable not less than
//         (n*(3*n+13))/2.
//
{
  double epsfcn;
  double factor = 100.0;
  int index;
  int info;
  int j;
  int lr;
  int maxfev;
  int ml;
  int mode;
  int mu;
  int nfev;
  int nprint;
  double xtol;

  info = 0;
//
//  Check the input.
//
  if ( n <= 0 )
  {
    return info;
  }
  if ( tol <= 0.0 )
  {
    return info;
  }
  if ( lwa < ( n * ( 3 * n + 13 ) ) / 2 )
  {
    return info;
  }
//
//  Call HYBRD.
//
  maxfev = 200 * ( n + 1 );
  xtol = tol;
  ml = n - 1;
  mu = n - 1;
  epsfcn = 0.0;
  mode = 2;
  for ( j = 0; j < n; j++ )
  {
    wa[j] = 1.0;
  }
  nprint = 0;
  lr = ( n * ( n + 1 ) ) / 2;
  index = 6 * n + lr;

  info = hybrd ( fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, wa, mode,
    factor, nprint, nfev, wa+index, n, wa+6*n, lr,
    wa+n, wa+2*n, wa+3*n, wa+4*n, wa+5*n );

  if ( info == 5 )
  {
    info = 4;
  }
  return info;
}
//****************************************************************************80

// int i4_max ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MAX returns the maximum of two I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 October 1998, John Burkardt
//    17 August 2013, David Goffredo
//        - Commented out for favor of std::max
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, are two objects to be compared.
//
//    Output, int I4_MAX, the larger of I1 and I2.
//
// {
//   int value;
//
//   if ( i2 < i1 )
//   {
//     value = i1;
//   }
//   else
//   {
//     value = i2;
//   }
//   return value;
// }
//****************************************************************************80

// int std::min ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MIN returns the minimum of two I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 October 1998, John Burkhardt
//    17 August 2013, David Goffredo
//        - Commented out for favor of std::min
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, two integers to be compared.
//
//    Output, int I4_MIN, the smaller of I1 and I2.
//
// {
//   int value;
// 
//   if ( i1 < i2 )
//   {
//     value = i1;
//   }
//   else
//   {
//     value = i2;
//   }
//   return value;
// }
//****************************************************************************80

void qform ( int m, int n, double q[], int ldq, double wa[] )

//****************************************************************************80
//
//  Purpose:
//
//    QFORM constructs the standard form of Q from its factored form.
//
//  Discussion:
//
//    This subroutine proceeds from the computed QR factorization of
//    an M by N matrix A to accumulate the M by M orthogonal matrix
//    Q from its factored form.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       m is a positive integer input variable set to the number
//         of rows of a and the order of q.
//
//       n is a positive integer input variable set to the number
//         of columns of a.
//
//       q is an m by m array. on input the full lower trapezoid in
//         the first min(m,n) columns of q contains the factored form.
//         on output q has been accumulated into a square matrix.
//
//       ldq is a positive integer input variable not less than m
//         which specifies the leading dimension of the array q.
//
//       wa is a work array of length m.
//
{
  int i;
  int j;
  int jm1;
  int k;
  int l;
  int minmn;
  int np1;
  double sum;
  double temp;
//
//  Zero out upper triangle of Q in the first min(M,N) columns.
//
  minmn = std::min ( m, n );

  for ( j = 1; j < minmn; j++ )
  {
    for ( i = 0; i <= j - 1; i++ )
    {
      q[i+j*ldq] = 0.0;
    }
  }
//
//  Initialize remaining columns to those of the identity matrix.
//
  for ( j = n; j < m; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      q[i+j*ldq] = 0.0;
    }
    q[j+j*ldq] = 1.0;
  }
//
//  Accumulate Q from its factored form.
//
  for ( k = minmn - 1; 0 <= k; k-- )
  {
    for ( i = k; i < m; i++ )
    {
      wa[i] = q[i+k*ldq];
      q[i+k*ldq] = 0.0;
    }
    q[k+k*ldq] = 1.0;

    if ( wa[k] != 0.0 )
    {
      for ( j = k; j < m; j++ )
      {
        sum = 0.0;
        for ( i = k; i < m; i++ )
        {
          sum +=  q[i+j*ldq] * wa[i];
        }
        temp = sum / wa[k];
        for ( i = k; i < m; i++ )
        {
          q[i+j*ldq] = q[i+j*ldq] - temp * wa[i];
        }
      }
    }
  }
}
//****************************************************************************80

void qrfac ( int m, int n, double a[], int lda, bool pivot, int ipvt[],
  int lipvt, double rdiag[], double acnorm[], double wa[] )

//****************************************************************************80
//
//  Purpose:
//
//    QRFAC computes the QR factorization of an M by N matrix.
//
//  Discussion:
//
//    this subroutine uses householder transformations with column
//    pivoting (optional) to compute a qr factorization of the
//    m by n matrix a. that is, qrfac determines an orthogonal
//    matrix q, a permutation matrix p, and an upper trapezoidal
//    matrix r with diagonal elements of nonincreasing magnitude,
//    such that a*p = q*r. the householder transformation for
//    column k, k = 1,2,...,min(m,n), is of the form
//
//      i - (1/u(k))*u*u'
//
//    where u has zeros in the first k-1 positions. the form of
//    this transformation and the method of pivoting first
//    appeared in the corresponding linpack subroutine.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       m is a positive integer input variable set to the number
//         of rows of a.
//
//       n is a positive integer input variable set to the number
//         of columns of a.
//
//       a is an m by n array. on input a contains the matrix for
//         which the qr factorization is to be computed. on output
//         the strict upper trapezoidal part of a contains the strict
//         upper trapezoidal part of r, and the lower trapezoidal
//         part of a contains a factored form of q (the non-trivial
//         elements of the u vectors described above).
//
//       lda is a positive integer input variable not less than m
//         which specifies the leading dimension of the array a.
//
//       pivot is a logical input variable. if pivot is set true,
//         then column pivoting is enforced. if pivot is set false,
//         then no column pivoting is done.
//
//       ipvt is an integer output array of length lipvt. ipvt
//         defines the permutation matrix 0. such that a*p = q*r.
//         column j of 0. is column ipvt(j) of the identity matrix.
//         if pivot is false, ipvt is not referenced.
//
//       lipvt is a positive integer input variable. if pivot is false,
//         then lipvt may be as small as 1. if pivot is true, then
//         lipvt must be at least n.
//
//       rdiag is an output array of length n which contains the
//         diagonal elements of r.
//
//       acnorm is an output array of length n which contains the
//         norms of the corresponding columns of the input matrix a.
//         if this information is not needed, then acnorm can coincide
//         with rdiag.
//
//       wa is a work array of length n. if pivot is false, then wa
//         can coincide with rdiag.
//
{
  double ajnorm;
  double epsmch;
  int i;
  int j;
  int jp1;
  int k;
  int kmax;
  int minmn;
  double sum;
  double temp;
//
//  EPSMCH is the machine precision.
//
  epsmch = std::numeric_limits<double>::epsilon();
//
//  Compute the initial column norms and initialize several arrays.
//
  for ( j = 0; j < n; j++ )
  {
    acnorm[j] = enorm ( m, a+j*lda );
    rdiag[j] = acnorm[j];
    wa[j] = rdiag[j];
    if ( pivot )
    {
      ipvt[j] = j;
    }
  }
//
//  Reduce A to R with Householder transformations.
//
  minmn = std::min ( m, n );

  for ( j = 0; j < minmn; j++ )
  {
    if ( pivot )
    {
//
//  Bring the column of largest norm into the pivot position.
//
      kmax = j;
      for ( k = j; k < n; k++ )
      {
        if ( rdiag[kmax] < rdiag[k] )
        {
          kmax = k;
        }
      }
      if ( kmax != j )
      {
        for ( i = 0; i < m; i++ )
        {
          temp          = a[i+j*lda];
          a[i+j*lda]    = a[i+kmax*lda];
          a[i+kmax*lda] = temp;
        }
        rdiag[kmax] = rdiag[j];
        wa[kmax] = wa[j];
        k          = ipvt[j];
        ipvt[j]    = ipvt[kmax];
        ipvt[kmax] = k;
      }
    }
//
//  Compute the Householder transformation to reduce the
//  J-th column of A to a multiple of the J-th unit vector.
//
    ajnorm = enorm ( m - j, a+j+j*lda );

    if ( ajnorm != 0.0 )
    {
      if ( a[j+j*lda] < 0.0 )
      {
        ajnorm = - ajnorm;
      }
      for ( i = j; i < m; i++ )
      {
        a[i+j*lda] = a[i+j*lda] / ajnorm;
      }
      a[j+j*lda] = a[j+j*lda] + 1.0;
//
//  Apply the transformation to the remaining columns and update the norms.
//
      jp1 = j + 1;
      for ( k = j + 1; k < n; k++ )
      {
        sum = 0.0;
        for ( i = j; i < m; i++ )
        {
          sum +=  a[i+j*lda] * a[i+k*lda];
        }
        temp = sum / a[j+j*lda];
        for ( i = j; i < m; i++ )
        {
          a[i+k*lda] -= temp * a[i+j*lda];
        }
        if ( pivot && rdiag[k] != 0.0 )
        {
          temp = a[j+k*lda] / rdiag[k];
          rdiag[k] = rdiag[k] * sqrt ( std::max ( 0.0, 1.0 - temp * temp ) );
          if ( 0.05 * ( rdiag[k] / wa[k] ) * ( rdiag[k] / wa[k] ) <= epsmch )
          {
            rdiag[k] = enorm ( m - 1 - j, a+(j+1)+k*lda );
            wa[k] = rdiag[k];
          }
        }
      }
    }
    rdiag[j] = - ajnorm;
  }
}
//****************************************************************************80

void r1mpyq ( int m, int n, double a[], int lda, double v[], double w[] )

//****************************************************************************80
//
//  Purpose:
//
//    R1MPYQ multiplies an M by N matrix A by the Q factor.
//
//  Discussion:
//
//    Given an m by n matrix a, this subroutine computes a*q where
//    q is the product of 2*(n - 1) transformations
//
//      gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)
//
//    and gv(i), gw(i) are givens rotations in the (i,n) plane which
//    eliminate elements in the i-th and n-th planes, respectively.
//    q itself is not given, rather the information to recover the
//    gv, gw rotations is supplied.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       m is a positive integer input variable set to the number
//         of rows of a.
//
//       n is a positive integer input variable set to the number
//         of columns of a.
//
//       a is an m by n array. on input a must contain the matrix
//         to be postmultiplied by the orthogonal matrix q
//         described above. on output a*q has replaced a.
//
//       lda is a positive integer input variable not less than m
//         which specifies the leading dimension of the array a.
//
//       v is an input array of length n. v(i) must contain the
//         information necessary to recover the givens rotation gv(i)
//         described above.
//
//       w is an input array of length n. w(i) must contain the
//         information necessary to recover the givens rotation gw(i)
//         described above.
//
{
  using std::sqrt;

  double c;
  int i;
  int j;
  double s;
  double temp;
//
//  Apply the first set of Givens rotations to A.
//
  for ( j = n - 2; 0 <= j; j-- )
  {
    if ( 1.0 < std::fabs ( v[j] ) )
    {
      c = 1.0 / v[j];
      s = sqrt ( 1.0 - c * c );
    }
    else
    {
      s = v[j];
      c = sqrt ( 1.0 - s * s );
    }
    for ( i = 0; i < m; i++ )
    {
      temp           = c * a[i+j*lda] - s * a[i+(n-1)*lda];
      a[i+(n-1)*lda] = s * a[i+j*lda] + c * a[i+(n-1)*lda];
      a[i+j*lda]     = temp;
    }
  }
//
//  Apply the second set of Givens rotations to A.
//
  for ( j = 0; j < n - 1; j++ )
  {
    if ( 1.0 < std::fabs ( w[j] ) )
    {
      c = 1.0 / w[j];
      s = sqrt ( 1.0 - c * c );
    }
    else
    {
      s = w[j];
      c = sqrt ( 1.0 - s * s );
    }
    for ( i = 0; i < m; i++ )
    {
      temp           =   c * a[i+j*lda] + s * a[i+(n-1)*lda];
      a[i+(n-1)*lda] = - s * a[i+j*lda] + c * a[i+(n-1)*lda];
      a[i+j*lda]     = temp;
    }
  }
}
//****************************************************************************80

bool r1updt ( int m, int n, double s[], int ls, double u[], double v[],
  double w[] )

//****************************************************************************80
//
//  Purpose:
//
//    R1UPDT updates the Q factor after a rank one update of the matrix.
//
//  Discussion:
//
//    Given an m by n lower trapezoidal matrix s, an m-vector u,
//    and an n-vector v, the problem is to determine an
//    orthogonal matrix q such that
//
//      (s + u*v') * q
//
//    is again lower trapezoidal.
//
//    This subroutine determines q as the product of 2*(n - 1)
//    transformations
//
//      gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)
//
//    where gv(i), gw(i) are givens rotations in the (i,n) plane
//    which eliminate elements in the i-th and n-th planes,
//    respectively. q itself is not accumulated, rather the
//    information to recover the gv, gw rotations is returned.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 April 2010
//
//  Author:
//
//    Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jorge More, Burton Garbow, Kenneth Hillstrom,
//    User Guide for MINPACK-1,
//    Technical Report ANL-80-74,
//    Argonne National Laboratory, 1980.
//
//  Parameters:
//
//       m is a positive integer input variable set to the number
//         of rows of s.
//
//       n is a positive integer input variable set to the number
//         of columns of s. n must not exceed m.
//
//       s is an array of length ls. on input s must contain the lower
//         trapezoidal matrix s stored by columns. on output s contains
//         the lower trapezoidal matrix produced as described above.
//
//       ls is a positive integer input variable not less than
//         (n*(2*m-n+1))/2.
//
//       u is an input array of length m which must contain the
//         vector u.
//
//       v is an array of length n. on input v must contain the vector
//         v. on output v(i) contains the information necessary to
//         recover the givens rotation gv(i) described above.
//
//       w is an output array of length m. w(i) contains information
//         necessary to recover the givens rotation gw(i) described
//         above.
//
//       sing is a logical output variable. sing is set true if any
//         of the diagonal elements of the output s are zero. otherwise
//         sing is set false.
//
{
  double cotan;
  double cs;
  double giant;
  int i;
  int j;
  int jj;
  int l;
  int nmj;
  int nm1;
  // double 0.25 = 0.25;
  // double 0.5 = 0.5;
  double sn;
  bool sing;
  double tan;
  double tau;
  double temp;
//
//  Because of the computation of the pointer JJ, this function was
//  converted from FORTRAN77 to C++ in a conservative way.  All computations
//  are the same, and only array indexing is adjusted.
//
//  GIANT is the largest magnitude.
//
  giant = r8_huge ( );
//
//  Initialize the diagonal element pointer.
//
  jj = ( n * ( 2 * m - n + 1 ) ) / 2 - ( m - n );
//
//  Move the nontrivial part of the last column of S into W.
//
  l = jj;
  for ( i = n; i <= m; i++ )
  {
    w[i-1] = s[l-1];
    l +=  1;
  }
//
//  Rotate the vector V into a multiple of the N-th unit vector
//  in such a way that a spike is introduced into W.
//
  nm1 = n - 1;

  for ( j = n - 1; 1 <= j; j-- )
  {
    jj -=  ( m - j + 1 );
    w[j-1] = 0.0;

    if ( v[j-1] != 0.0 )
    {
//
//  Determine a Givens rotation which eliminates the J-th element of V.
//
      if ( std::fabs ( v[n-1] ) < std::fabs ( v[j-1] ) )
      {
        cotan = v[n-1] / v[j-1];
        sn = 0.5 / sqrt ( 0.25 + 0.25 * cotan * cotan );
        cs = sn * cotan;
        tau = 1.0;
        if ( 1.0 < std::fabs ( cs ) * giant )
        {
          tau = 1.0 / cs;
        }
      }
      else
      {
        tan = v[j-1] / v[n-1];
        cs = 0.5 / sqrt ( 0.25 + 0.25 * tan * tan );
        sn = cs * tan;
        tau = sn;
      }
//
//  Apply the transformation to V and store the information
//  necessary to recover the Givens rotation.
//
      v[n-1] = sn * v[j-1] + cs * v[n-1];
      v[j-1] = tau;
//
//  Apply the transformation to S and extend the spike in W.
//
      l = jj;
      for ( i = j; i <= m; i++ )
      {
        temp   = cs * s[l-1] - sn * w[i-1];
        w[i-1] = sn * s[l-1] + cs * w[i-1];
        s[l-1] = temp;
        l +=  1;
      }
    }
  }
//
//  Add the spike from the rank 1 update to W.
//
  for ( i = 1; i <= m; i++ )
  {
     w[i-1] = w[i-1] + v[n-1] * u[i-1];
  }
//
//  Eliminate the spike.
//
  sing = false;

  for ( j = 1; j <= nm1; j++ )
  {
//
//  Determine a Givens rotation which eliminates the
//  J-th element of the spike.
//
    if ( w[j-1] != 0.0 )
    {

      if ( std::fabs ( s[jj-1] ) < std::fabs ( w[j-1] ) )
      {
        cotan = s[jj-1] / w[j-1];
        sn = 0.5 / sqrt ( 0.25 + 0.25 * cotan * cotan );
        cs = sn * cotan;
        tau = 1.0;
        if ( 1.0 < std::fabs ( cs ) * giant )
        {
          tau = 1.0 / cs;
        }
      }
      else
      {
        tan = w[j-1] / s[jj-1];
        cs = 0.5 / sqrt ( 0.25 + 0.25 * tan * tan );
        sn = cs * tan;
        tau = sn;
      }
//
//  Apply the transformation to s and reduce the spike in w.
//
      l = jj;

      for ( i = j; i <= m; i++ )
      {
        temp   =   cs * s[l-1] + sn * w[i-1];
        w[i-1] = - sn * s[l-1] + cs * w[i-1];
        s[l-1] = temp;
        l +=  1;
      }
//
//  Store the information necessary to recover the givens rotation.
//
      w[j-1] = tau;
    }
//
//  Test for zero diagonal elements in the output s.
//
    if ( s[jj-1] == 0.0 )
    {
      sing = true;
    }
    jj +=  ( m - j + 1 );
  }
//
//  Move W back into the last column of the output S.
//
  l = jj;
  for ( i = n; i <= m; i++ )
  {
    s[l-1] = w[i-1];
    l +=  1;
  }
  if ( s[jj-1] == 0.0 )
  {
    sing = true;
  }
  return sing;
}
//****************************************************************************80

// double r8_abs ( double x )

//****************************************************************************80
//
//  Purpose:
//
//    R8_ABS returns the absolute value of an R8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 November 2006
//    17 August 2013, David Goffredo
//        - Commented out for favor of std::fabs
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the quantity whose absolute value is desired.
//
//    Output, double R8_ABS, the absolute value of X.
//
// {
//   double value;
// 
//   if ( 0.0 <= x )
//   {
//     value = x;
//   }
//   else
//   {
//     value = - x;
//   }
//   return value;
// }
//****************************************************************************80

// double r8_epsilon ( )

//****************************************************************************80
//
//  Purpose:
//
//    R8_EPSILON returns the R8 roundoff unit.
//
//  Discussion:
//
//    The roundoff unit is a number R which is a power of 2 with the
//    property that, to the precision of the computer's arithmetic,
//      1 < 1 + R
//    but
//      1 = ( 1 + R / 2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    01 July 2004, John Burkhardt
//    17 August 2013, David Goffredo
//        - Commented out for favor of std::numeric_limits<double>::epsilon()
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_EPSILON, the R8 round-off unit.
//
// {
//   double value;
// 
//   value = 1.0;
// 
//   while ( 1.0 < ( double ) ( 1.0 + value )  )
//   {
//     value /=  2.0;
//   }
// 
//   value = 2.0 * value;
// 
//   return value;
// }
//****************************************************************************80

double r8_huge ( )

//****************************************************************************80
//
//  Purpose:
//
//    R8_HUGE returns a "huge" R8.
//
//  Discussion:
//
//    The value returned by this function is NOT required to be the
//    maximum representable R8.  This value varies from machine to machine,
//    from compiler to compiler, and may cause problems when being printed.
//    We simply want a "very large" but non-infinite number.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    06 October 2007
//    17 August 2013, David Goffredo
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_HUGE, a "huge" R8 value.
//
{
  return 1.0E+30;
}
//****************************************************************************80

// double r8_max ( double x, double y )

//****************************************************************************80
//
//  Purpose:
//
//    R8_MAX returns the maximum of two R8's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    18 August 2004
//    17 August 2013
//        - Commented out for favor of std::max
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, Y, the quantities to compare.
//
//    Output, double R8_MAX, the maximum of X and Y.
//
// {
//   double value;
// 
//   if ( y < x )
//   {
//     value = x;
//   }
//   else
//   {
//     value = y;
//   }
//   return value;
// }
//****************************************************************************80

// double r8_min ( double x, double y )

//****************************************************************************80
//
//  Purpose:
//
//    R8_MIN returns the minimum of two R8's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    31 August 2004
//    17 August 2013, David Goffredo
//        - Commented out for favor of std::min
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, Y, the quantities to compare.
//
//    Output, double R8_MIN, the minimum of X and Y.
//
// {
//   double value;
// 
//   if ( y < x )
//   {
//     value = y;
//   }
//   else
//   {
//     value = x;
//   }
//   return value;
// }
//****************************************************************************80

// double r8_tiny ( )

//****************************************************************************80
//
//  Purpose:
//
//    R8_TINY returns a "tiny" R8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 March 2007
//    17 August 2013, David Goffredo
//        - Commented out since it's unused.
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_TINY, a "tiny" R8 value.
//
// {
//   return 0.4450147717014E-307; // std::numeric_limits<double>::min() instead?
// }
//****************************************************************************80

double r8_uniform_01 ( int *seed ) // std::rand() instead?
                                   // Nevermind, this is unused anyway.

//****************************************************************************80
//
//  Purpose:
//
//    R8_UNIFORM_01 returns a unit pseudorandom R8.
//
//  Discussion:
//
//    This routine implements the recursion
//
//      seed = ( 16807 * seed ) mod ( 2^31 - 1 )
//      u = seed / ( 2^31 - 1 )
//
//    The integer arithmetic never requires more than 32 bits,
//    including a sign bit.
//
//    If the initial seed is 12345, then the first three computations are
//
//      Input     Output      R8_UNIFORM_01
//      SEED      SEED
//
//         12345   207482415  0.096616
//     207482415  1790989824  0.833995
//    1790989824  2035175616  0.947702
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Paul Bratley, Bennett Fox, Linus Schrage,
//    A Guide to Simulation,
//    Second Edition,
//    Springer, 1987,
//    ISBN: 0387964673,
//    LC: QA76.9.C65.B73.
//
//    Bennett Fox,
//    Algorithm 647:
//    Implementation and Relative Efficiency of Quasirandom
//    Sequence Generators,
//    ACM Transactions on Mathematical Software,
//    Volume 12, Number 4, December 1986, pages 362-376.
//
//    Pierre L'Ecuyer,
//    Random Number Generation,
//    in Handbook of Simulation,
//    edited by Jerry Banks,
//    Wiley, 1998,
//    ISBN: 0471134031,
//    LC: T57.62.H37.
//
//    Peter Lewis, Allen Goodman, James Miller,
//    A Pseudo-Random Number Generator for the System/360,
//    IBM Systems Journal,
//    Volume 8, Number 2, 1969, pages 136-143.
//
//  Parameters:
//
//    Input/output, int *SEED, the "seed" value.  Normally, this
//    value should not be 0.  On output, SEED has been updated.
//
//    Output, double R8_UNIFORM_01, a new pseudorandom variate,
//    strictly between 0 and 1.
//
{
  int i4_huge = 2147483647;
  int k;
  double r;

  if ( *seed == 0 )
  {
    std::cerr << "\n";
    std::cerr << "R8_UNIFORM_01 - Fatal error!\n";
    std::cerr << "  Input value of SEED = 0.\n";
    exit ( 1 );
  }

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }
//
//  Although SEED can be represented exactly as a 32 bit integer,
//  it generally cannot be represented exactly as a 32 bit real number.
//
  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}
//****************************************************************************80

void r8mat_print ( int m, int n, double a[], std::string title )

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_PRINT prints an R8MAT.
//
//  Discussion:
//
//    An R8MAT is a doubly dimensioned array of R8 values,  stored as a vector
//    in column-major order.
//
//    Entry A(I,J) is stored as A[I+J*M]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    10 September 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int M, the number of rows in A.
//
//    Input, int N, the number of columns in A.
//
//    Input, double A[M*N], the M by N matrix.
//
//    Input, std::string TITLE, a title.
//
{
  r8mat_print_some ( m, n, a, 1, 1, m, n, title );
}
//****************************************************************************80

void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi,
  int jhi, std::string title )

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_PRINT_SOME prints some of an R8MAT.
//
//  Discussion:
//
//    An R8MAT is a doubly dimensioned array of R8 values,  stored as a vector
//    in column-major order.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    10 September 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int M, the number of rows of the matrix.
//    M must be positive.
//
//    Input, int N, the number of columns of the matrix.
//    N must be positive.
//
//    Input, double A[M*N], the matrix.
//
//    Input, int ILO, JLO, IHI, JHI, designate the first row and
//    column, and the last row and column to be printed.
//
//    Input, std::string TITLE, a title.
//
{
  enum { INCX = 5 };

  int i;
  int i2hi;
  int i2lo;
  int j;
  int j2hi;
  int j2lo;

  std::cout << "\n";
  std::cout << title << "\n";
//
//  Print the columns of the matrix, in strips of 5.
//
  for ( j2lo = jlo; j2lo <= jhi; j2lo +=  INCX )
  {
    j2hi = j2lo + INCX - 1;
    j2hi = std::min ( j2hi, n );
    j2hi = std::min ( j2hi, jhi );

    std::cout << "\n";
//
//  For each column J in the current range...
//
//  Write the header.
//
    std::cout << "  Col:    ";
    for ( j = j2lo; j <= j2hi; j++ )
    {
      std::cout << std::setw(7) << j << "       ";
    }
    std::cout << "\n";
    std::cout << "  Row\n";
    std::cout << "\n";
//
//  Determine the range of the rows in this strip.
//
    i2lo = std::max ( ilo, 1 );
    i2hi = std::min ( ihi, m );

    for ( i = i2lo; i <= i2hi; i++ )
    {
//
//  Print out (up to) 5 entries in row I, that lie in the current strip.
//
      std::cout << std::setw(5) << i << "  ";
      for ( j = j2lo; j <= j2hi; j++ )
      {
        std::cout << std::setw(12) << a[i-1+(j-1)*m] << "  ";
      }
      std::cout << "\n";
    }
  }
}
//****************************************************************************80

void r8vec_print ( int n, double a[], std::string title )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_PRINT prints an R8VEC.
//
//  Discussion:
//
//    An R8VEC is a vector of R8's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    16 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of components of the vector.
//
//    Input, double A[N], the vector to be printed.
//
//    Input, std::string TITLE, a title.
//
{
  int i;

  std::cout << "\n";
  std::cout << title << "\n";
  std::cout << "\n";
  for ( i = 0; i < n; i++ )
  {
    std::cout << "  " << std::setw(8)  << i
         << "  " << std::setw(14) << a[i]  << "\n";
  }
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
  enum { TIME_SIZE = 40 };

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  size_t len;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  len = std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

  std::cout << time_buffer << "\n";
}

// enum { nVars = 32 };
// #define ZERO_CONSTRAINT(index, expression) fvec[index] = (expression) - x[index]
// // void fcn ( int n, double x[], double fvec[], int *iflag )
// void test1(int n, double x[], double fvec[], int *iflag)
// {
//     // assert(n == nVars)
// /* Small example
//     ZERO_CONSTRAINT(0, 1 + x[4]);
//     ZERO_CONSTRAINT(1, x[5] + x[0]);
//     ZERO_CONSTRAINT(2, x[6] + x[1]);
//     ZERO_CONSTRAINT(3, 1 + x[7]);
//     ZERO_CONSTRAINT(4, 1.0);
//     ZERO_CONSTRAINT(5, 1.0);
//     double denom = 1.0 / (1.0 / x[1] + 1.0);
//     ZERO_CONSTRAINT(6, (1.0 / x[1]) * denom);
//     ZERO_CONSTRAINT(7, denom);
// */
// 
//     // /* This is for a more complicated case.
//     enum { nInverses = nVars / 2 };
//     double inverses[nInverses];
// 
//     for(unsigned ii = 0; ii < nInverses; ++ii)
//         inverses[ii] = 1.0 / x[ii];
// 
//     enum { nTickets = 5 };
//     // (w)ait (c)ontribution (d)enominators
//     double wcd[nTickets] = { 1.0 / (1 + inverses[3] + inverses[9] + inverses[10] + inverses[13]), // 0 = triangle
//                              1.0 / (inverses[0] + 1.0 + inverses[7] + inverses[14]),              // 1 = square
//                              1.0 / (inverses[1] + inverses[5] + 2.0),                             // 2 = circle
//                              1.0 / (inverses[2] + inverses[4] + inverses[11] + inverses[12]),     // 3 = dot
//                              1.0 / (inverses[6] + inverses[8] + 1.0 + inverses[15]) };            // 4 = tilde
// 
//     ZERO_CONSTRAINT(0, 1.0 + x[16]);
//     ZERO_CONSTRAINT(1, x[17] + x[0]);
//     ZERO_CONSTRAINT(2, x[18] + x[1]);
//     ZERO_CONSTRAINT(3, 1.0 + x[19]);
//     ZERO_CONSTRAINT(4, x[20] + x[3]);
//     ZERO_CONSTRAINT(5, x[21] + x[4]);
//     ZERO_CONSTRAINT(6, x[22] + x[5]);
//     ZERO_CONSTRAINT(7, 1.0 + x[23]);
//     ZERO_CONSTRAINT(8, x[24] + x[7]);
//     ZERO_CONSTRAINT(9, x[25] + x[8]);
//     ZERO_CONSTRAINT(10, 1.0 + x[26]);
//     ZERO_CONSTRAINT(11, x[27] + x[10]);
//     ZERO_CONSTRAINT(12, 1.0 + x[28]);
//     ZERO_CONSTRAINT(13, x[29] + x[12]);
//     ZERO_CONSTRAINT(14, x[30] + x[13]);
//     ZERO_CONSTRAINT(15, x[31] + x[14]);
// 
//     ZERO_CONSTRAINT(16, wcd[0]);
//     ZERO_CONSTRAINT(17, inverses[0] * wcd[1]);
//     ZERO_CONSTRAINT(18, inverses[1] * wcd[2]);
//     ZERO_CONSTRAINT(19, wcd[1]);
//     ZERO_CONSTRAINT(20, inverses[3] * wcd[0]);
//     ZERO_CONSTRAINT(21, inverses[4] * wcd[3]);
//     ZERO_CONSTRAINT(22, inverses[5] * wcd[2]);
//     ZERO_CONSTRAINT(23, wcd[2]);
//     ZERO_CONSTRAINT(24, inverses[7] * wcd[1]);
//     ZERO_CONSTRAINT(25, inverses[8] * wcd[4]);
//     ZERO_CONSTRAINT(26, wcd[4]);
//     ZERO_CONSTRAINT(27, inverses[10] * wcd[0]);
//     ZERO_CONSTRAINT(28, wcd[2]);
//     ZERO_CONSTRAINT(29, inverses[12] * wcd[3]);
//     ZERO_CONSTRAINT(30, inverses[13] * wcd[0]);
//     ZERO_CONSTRAINT(31, inverses[14] * wcd[1]);
//     // */
// 
//     // This is for the simplest cyclic mode
//     // x is [ wt(t, 1) = 1 (excluded), 
//     //        0. wt(s, 1), 
//     //        wt(s, 2) = 1 (excluded), 
//     //        1. wt(t, 2), 
//     //        2. wc(t, 1), 
//     //        3. wc(s, 1) ] 
//     // 
//     // fvec[0] = 1 + x[2] - x[0];
//     // fvec[1] = 1 + x[3] - x[1];
//     //
//     // fvec[2] = 1.0 / (1.0 + 1.0 / x[1]) - x[2];
//     //
//     // fvec[3] = 1.0 / (1.0 + 1.0 / x[0]) - x[3];
// 
// /* Printing iterates
//     for (unsigned ii = 0; ii < n; ++ii)
//     {
//         std::cout << "x[" << ii << "] = " << x[ii] << "    "
//                   << "f[" << ii << "] = " << fvec[ii] << '\n';
//     }
// */
// }

template <typename Func>
// Func has ' /* unused */ operator() (const double inputs[],
//                                     double outputs[])
class CallbackAdaptor
{
    Func & _func;
    std::vector<double> _inputs;
    std::vector<double> _outputs;
    std::vector<double> _work;
  public:

    CallbackAdaptor(Func & func, std::vector<double> initialGuess)
        : _func(func), _outputs(initialGuess.size()),
          _work((initialGuess.size() * (3 * initialGuess.size() + 13)) / 2 + 1)
    {
        using std::swap;
        swap(_inputs, initialGuess);
    }

    double * inputs() { return &_inputs.front(); }
    double * outputs() { return &_outputs.front(); }
    double * work() { return &_work.front(); }

    const std::vector<double> lastArgument() const { return _inputs; }
    const std::vector<double> lastIterate() const { return _outputs; }

    int nVars() const { return _inputs.size(); }
    int workSize() const { return _work.size(); }

    void operator ()(int n, double x[], double fvec[], int *iflag)
    {
        (void)iflag; // Unused: We don't want to halt prematurely yet.
        (void)n;     // Unused: _func knows the size of the arrays.

        _func(static_cast<const double*>(x), fvec);
    }
};

template <typename Adaptor>
struct CallbackRef
{
    Adaptor & _adaptor;

    CallbackRef(Adaptor & adaptor)
        : _adaptor(adaptor) {}

    void operator ()(int n, double x[], double fvec[], int *iflag)
    {
        _adaptor(n, x, fvec, iflag);
    }
};

template <typename Adaptor>
CallbackRef<Adaptor> callback_ref(Adaptor & adaptor)
{
    return CallbackRef<Adaptor>(adaptor);
}

// // Example usage:
//
// std::vector<double> initialGuess = getInitialGuess();
//
// MySystem system;
//
// CallbackAdaptor<MySystem> adaptor(system, initialGuess);
//
// int result = hybrd1(callback_ref(adaptor), // fcn 
//                     adaptor.nVars(),       // n
//                     adaptor.inputs(),      // x
//                     adaptor.outputs(),     // fvec
//                     0.0001,                // tol
//                     adaptor.work(),        // wa
//                     adaptor.workSize());   // lwa
//

struct SimpleSystem
{
    enum { nVars = 32,
           nInverses = nVars / 2,
           nTickets = 5, };

    double inverses[nInverses];
    double wcd[nTickets];

    std::vector<double> guess() const
    {
        return std::vector<double>(nVars, 1.0);
    }

    void operator() (const double x[], double fvec[])
    {
        // double f[nVars];

        for(unsigned ii = 0; ii < nInverses; ++ii)
            inverses[ii] = 1.0 / x[ii];

        wcd[0] = 1.0 / (1 + inverses[3] + inverses[9] + inverses[10] + inverses[13]); // 0 = triangle
        wcd[1] = 1.0 / (inverses[0] + 1.0 + inverses[7] + inverses[14]);              // 1 = square
        wcd[2] = 1.0 / (inverses[1] + inverses[5] + 2.0);                             // 2 = circle
        wcd[3] = 1.0 / (inverses[2] + inverses[4] + inverses[11] + inverses[12]);     // 3 = dot
        wcd[4] = 1.0 / (inverses[6] + inverses[8] + 1.0 + inverses[15]);              // 4 = tilde

        #define ZERO_CONSTRAINT(index, expression) \
            fvec[index] = (expression) - x[index]
        ZERO_CONSTRAINT(0, 1.0 + x[16]);
        ZERO_CONSTRAINT(1, x[17] + x[0]);
        ZERO_CONSTRAINT(2, x[18] + x[1]);
        ZERO_CONSTRAINT(3, 1.0 + x[19]);
        ZERO_CONSTRAINT(4, x[20] + x[3]);
        ZERO_CONSTRAINT(5, x[21] + x[4]);
        ZERO_CONSTRAINT(6, x[22] + x[5]);
        ZERO_CONSTRAINT(7, 1.0 + x[23]);
        ZERO_CONSTRAINT(8, x[24] + x[7]);
        ZERO_CONSTRAINT(9, x[25] + x[8]);
        ZERO_CONSTRAINT(10, 1.0 + x[26]);
        ZERO_CONSTRAINT(11, x[27] + x[10]);
        ZERO_CONSTRAINT(12, 1.0 + x[28]);
        ZERO_CONSTRAINT(13, x[29] + x[12]);
        ZERO_CONSTRAINT(14, x[30] + x[13]);
        ZERO_CONSTRAINT(15, x[31] + x[14]);

        ZERO_CONSTRAINT(16, wcd[0]);
        ZERO_CONSTRAINT(17, inverses[0] * wcd[1]);
        ZERO_CONSTRAINT(18, inverses[1] * wcd[2]);
        ZERO_CONSTRAINT(19, wcd[1]);
        ZERO_CONSTRAINT(20, inverses[3] * wcd[0]);
        ZERO_CONSTRAINT(21, inverses[4] * wcd[3]);
        ZERO_CONSTRAINT(22, inverses[5] * wcd[2]);
        ZERO_CONSTRAINT(23, wcd[2]);
        ZERO_CONSTRAINT(24, inverses[7] * wcd[1]);
        ZERO_CONSTRAINT(25, inverses[8] * wcd[4]);
        ZERO_CONSTRAINT(26, wcd[4]);
        ZERO_CONSTRAINT(27, inverses[10] * wcd[0]);
        ZERO_CONSTRAINT(28, wcd[2]);
        ZERO_CONSTRAINT(29, inverses[12] * wcd[3]);
        ZERO_CONSTRAINT(30, inverses[13] * wcd[0]);
        ZERO_CONSTRAINT(31, inverses[14] * wcd[1]);
        #undef ZERO_CONSTRAINT

        // std::copy(f, f + nVars, fvec);
    }
};

struct NoWcSystem
{
    enum { nVars = 16,
           nTickets = 5 };

    double inverses[nVars];
    double wcd[nTickets];

    std::vector<double> guess() const
    {
        return std::vector<double>(nVars, 1.0);
    }

    void operator() (const double x[], double fvec[])
    {
        // double f[nVars];

        for(unsigned ii = 0; ii < nVars; ++ii)
            inverses[ii] = 1.0 / x[ii];

        // (w)ait (c)ontribution (d)enominators
        wcd[0] = 1.0 / (1 + inverses[3] + inverses[9] + inverses[10] + inverses[13]); // 0 = triangle
        wcd[1] = 1.0 / (inverses[0] + 1.0 + inverses[7] + inverses[14]);              // 1 = square
        wcd[2] = 1.0 / (inverses[1] + inverses[5] + 2.0);                             // 2 = circle
        wcd[3] = 1.0 / (inverses[2] + inverses[4] + inverses[11] + inverses[12]);     // 3 = dot
        wcd[4] = 1.0 / (inverses[6] + inverses[8] + 1.0 + inverses[15]);              // 4 = tilde

        #define ZERO_CONSTRAINT(index, expression) \
            fvec[index] = (expression) - x[index]
        ZERO_CONSTRAINT(0, 1.0 + (wcd[0]));
        ZERO_CONSTRAINT(1, (inverses[0] * wcd[1]) + x[0]);
        ZERO_CONSTRAINT(2, (inverses[1] * wcd[2]) + x[1]);
        ZERO_CONSTRAINT(3, 1.0 + wcd[1]);
        ZERO_CONSTRAINT(4, (inverses[3] * wcd[0]) + x[3]);
        ZERO_CONSTRAINT(5, (inverses[4] * wcd[3]) + x[4]);
        ZERO_CONSTRAINT(6, (inverses[5] * wcd[2]) + x[5]);
        ZERO_CONSTRAINT(7, 1.0 + wcd[2]);
        ZERO_CONSTRAINT(8, (inverses[7] * wcd[1]) + x[7]);
        ZERO_CONSTRAINT(9, (inverses[8] * wcd[4]) + x[8]);
        ZERO_CONSTRAINT(10, 1.0 + wcd[4]);
        ZERO_CONSTRAINT(11, (inverses[10] * wcd[0]) + x[10]);
        ZERO_CONSTRAINT(12, 1.0 + wcd[2]);
        ZERO_CONSTRAINT(13, (inverses[12] * wcd[3]) + x[12]);
        ZERO_CONSTRAINT(14, (inverses[13] * wcd[0]) + x[13]);
        ZERO_CONSTRAINT(15, (inverses[14] * wcd[1]) + x[14]);
        #undef ZERO_CONSTRAINT

        // std::copy(f, f + nVars, fvec);
    }
};


class HybrdException : public std::exception
{
  public:
    enum { nErrors = 5,
           success = 1 };

    explicit HybrdException(int rcode) : _rcode(rcode) {}
    ~HybrdException() throw() {}
    
    const char* what() const throw() { return messages[_rcode]; }
  
  private:
    static const char* messages[nErrors];
    int _rcode;
};

const char* HybrdException::messages[HybrdException::nErrors] = {
    "Improper input parameters",
    "Success: algorithm estimates that the relative error between x and the solution is at most tol.",
    "Calculation is taking too many iterations. Number of calls to fcn has reached or exceeded 200*(n+1)",
    "Inter-iterate relative error stopping threshold is too small. No further improvement is possible.",
    "Iteration is not making good progress. Check initial conditions or the iterate function."
};

template <typename System>
std::vector<double> findZeroes(System & system, 
                               const std::vector<double> & initialGuess,
                               double stoppingRelativeError = 0.0001)
{
    CallbackAdaptor<System> adaptor(system, initialGuess);

//     std::cout << "Input parameters\n"
//               << "\tguess size = " << initialGuess.size() << '\n'
//               << "\tnVars = " << adaptor.nVars() << '\n'
//               << "\tinputs = " << adaptor.inputs() << '\n'
//               << "\toutputs = " << adaptor.outputs() << '\n'
//               << "\twork = " << adaptor.work() << '\n'
//               << "\tworkSize = " << adaptor.workSize() << '\n';

    int result = hybrd1(callback_ref(adaptor),
                        adaptor.nVars(),
                        adaptor.inputs(),
                        adaptor.outputs(),
                        stoppingRelativeError,
                        adaptor.work(),
                        adaptor.workSize());

    if (result != HybrdException::success)
        throw HybrdException(result);

    return adaptor.lastArgument();
}

int main()
{
    enum { nTrials = 1000 };
    std::vector<std::clock_t> trials;
    for (unsigned trial = 0; trial < nTrials; ++trial)
    {
        std::clock_t begin = std::clock();
        try 
        {
            // SimpleSystem system;
            NoWcSystem system;
            std::vector<double> result =  findZeroes(system, system.guess());

            std::system("sleep 0.01");
            std::clock_t end = std::clock();

            // for (unsigned ii = 0; ii < result.size(); ++ii)
            // {
            //     std::cout << "x[" << ii << "] = " << result[ii] << '\n';
            // }

            trials.push_back(end - begin);
        }
        catch (const std::exception & ex)
        {
            std::cerr << "Bad simulation. " << ex.what() << '\n';
        }
    }

    for (unsigned ii = 0; ii < trials.size(); ++ii)
        std::cout << trials[ii] << ' ';
    std::cout << '\n';

    double meanTime = (double)std::accumulate(trials.begin(), trials.end(), 0) / trials.size();
    std::cout << "Mean seconds taken was " << (meanTime / CLOCKS_PER_SEC) << '\n';
}


