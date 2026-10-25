! Reference driver for Hairer's DOPRI5 / DOP853 (golden vectors for the Rust
! embedded Runge-Kutta tests). Compile once per method, because both Fortran
! files define a function HINIT:
!
!   gfortran -cpp -DMETHOD_DOPRI5 reference_driver.F90 dopri5.f
!   gfortran -cpp -DMETHOD_DOP853 reference_driver.F90 dop853.f
!
! usage: reference_driver <case>     (writes TSV to standard output)

module reference_state
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer :: problem = 0           ! 1 = Arenstorf, 2 = Prothero-Robinson
  integer :: dimension_n = 0
  real(dp), parameter :: arenstorf_mu = 0.012277471d0
  real(dp), parameter :: arenstorf_x_end = 17.0652165601579625588917206249d0
  real(dp), parameter :: arenstorf_y2_dot_0 = -2.00158510637908252240537862224d0
contains
  ! 17 significant digits: enough for every double to round-trip.
  function number(x) result(text)
    real(dp), intent(in) :: x
    character(len=32) :: text
    write(text, '(ES25.16E3)') x
    text = adjustl(text)
  end function number

  function integer_text(i) result(text)
    integer, intent(in) :: i
    character(len=24) :: text
    write(text, '(I0)') i
  end function integer_text
end module reference_state

subroutine right_hand_side(n, x, y, f, rpar, ipar)
  use reference_state
  implicit none
  integer, intent(in) :: n
  real(dp), intent(in) :: x, y(n)
  real(dp), intent(out) :: f(n)
  real(dp) :: rpar(*)
  integer :: ipar(*)
  real(dp) :: mu, mu_prime, r1, r2

  if (problem == 1) then
    mu = arenstorf_mu
    mu_prime = 1.0d0 - mu
    r1 = (y(1) + mu)*(y(1) + mu) + y(2)*y(2)
    r1 = r1*sqrt(r1)
    r2 = (y(1) - mu_prime)*(y(1) - mu_prime) + y(2)*y(2)
    r2 = r2*sqrt(r2)
    f(1) = y(3)
    f(2) = y(4)
    f(3) = (((y(1) + 2.0d0*y(4)) - (mu_prime*(y(1) + mu))/r1) &
            - (mu*(y(1) - mu_prime))/r2)
    f(4) = (((y(2) - 2.0d0*y(3)) - (mu_prime*y(2))/r1) - (mu*y(2))/r2)
  else
    f(1) = (-1000.0d0*(y(1) - cos(x))) - sin(x)
  end if
end subroutine right_hand_side

#ifdef METHOD_DOP853
subroutine solution_output(nr, xold, x, y, n, con, icomp, nd, rpar, ipar, &
                           irtrn, xout)
#else
subroutine solution_output(nr, xold, x, y, n, con, icomp, nd, rpar, ipar, &
                           irtrn)
#endif
  use reference_state
  implicit none
  integer, intent(in) :: nr, n, nd
  real(dp), intent(in) :: xold, x, y(n), con(*)
  integer, intent(in) :: icomp(nd)
  real(dp) :: rpar(*)
  integer :: ipar(*), irtrn
#ifdef METHOD_DOP853
  real(dp) :: xout
  real(dp), external :: contd8
#else
  real(dp), external :: contd5
#endif
  real(dp), parameter :: thetas(3) = [0.25d0, 0.5d0, 0.75d0]
  character(len=4096) :: line
  real(dp) :: x_theta, value
  integer :: i, k

  line = 'step' // achar(9) // trim(integer_text(nr)) // achar(9) // &
         trim(number(xold)) // achar(9) // trim(number(x))
  do i = 1, n
    line = trim(line) // achar(9) // trim(number(y(i)))
  end do
  write(*, '(A)') trim(line)

  if (nr >= 2) then
    do k = 1, 3
      x_theta = xold + thetas(k)*(x - xold)
      line = 'dense' // achar(9) // trim(integer_text(nr)) // achar(9) // &
             trim(number(thetas(k))) // achar(9) // trim(number(x_theta))
      do i = 1, n
#ifdef METHOD_DOP853
        value = contd8(i, x_theta, con, icomp, nd)
#else
        value = contd5(i, x_theta, con, icomp, nd)
#endif
        line = trim(line) // achar(9) // trim(number(value))
      end do
      write(*, '(A)') trim(line)
    end do
  end if
  irtrn = 1
end subroutine solution_output

program reference_driver
  use reference_state
  implicit none
  external right_hand_side, solution_output
  integer, parameter :: n_max = 4
#ifdef METHOD_DOP853
  integer, parameter :: lwork = 11*n_max + 8*n_max + 21
  character(len=*), parameter :: method = 'dop853'
#else
  integer, parameter :: lwork = 8*n_max + 5*n_max + 21
  character(len=*), parameter :: method = 'dopri5'
#endif
  integer, parameter :: liwork = n_max + 21
  real(dp) :: work(lwork), y(n_max), rtol(n_max), atol(n_max), x, x_end
  integer :: iwork(liwork), ipar(1), itol, iout, idid, i
  real(dp) :: rpar(1)
  character(len=64) :: case_name
  character(len=4096) :: line

  call get_command_argument(1, case_name)
  work = 0.0d0
  iwork = 0
  rtol = 0.0d0
  atol = 0.0d0
  itol = 0
  iout = 2
  work(1) = epsilon(1.0d0)
  iwork(3) = -1

  select case (trim(case_name))
  case ('arenstorf_tol1e-4', 'arenstorf_tol1e-7', 'arenstorf_tol1e-10', &
        'arenstorf_backward_tol1e-7', 'arenstorf_componentwise', &
        'arenstorf_given_step')
    problem = 1
    dimension_n = 4
    y(1:4) = [0.994d0, 0.0d0, 0.0d0, arenstorf_y2_dot_0]
    x = 0.0d0
    x_end = arenstorf_x_end
    select case (trim(case_name))
    case ('arenstorf_tol1e-4')
      rtol(1) = 1.0d-4
    case ('arenstorf_tol1e-7')
      rtol(1) = 1.0d-7
    case ('arenstorf_tol1e-10')
      rtol(1) = 1.0d-10
    case ('arenstorf_backward_tol1e-7')
      rtol(1) = 1.0d-7
      x = arenstorf_x_end
      x_end = 0.0d0
    case ('arenstorf_componentwise')
      itol = 1
      atol(1:4) = [1.0d-8, 1.0d-8, 1.0d-6, 1.0d-6]
      rtol(1:4) = [1.0d-7, 1.0d-7, 1.0d-5, 1.0d-5]
    case ('arenstorf_given_step')
      rtol(1) = 1.0d-7
      work(7) = 1.0d-3
      work(6) = 0.5d0
    end select
    if (itol == 0) atol(1) = rtol(1)
  case ('prothero_robinson_stiff')
    problem = 2
    dimension_n = 1
    y(1) = 1.0d0
    x = 0.0d0
    x_end = 10.0d0
    rtol(1) = 1.0d-6
    atol(1) = 1.0d-6
  case default
    write(0, '(A)') 'unknown case: ' // trim(case_name)
    stop 2
  end select
  iwork(5) = dimension_n

  write(*, '(A)') '# method' // achar(9) // method
  write(*, '(A)') '# case' // achar(9) // trim(case_name)
  write(*, '(A)') '# n' // achar(9) // trim(integer_text(dimension_n))
  write(*, '(A)') '# x0' // achar(9) // trim(number(x)) // achar(9) // &
                  'x_end' // achar(9) // trim(number(x_end))
  line = '# y0'
  do i = 1, dimension_n
    line = trim(line) // achar(9) // trim(number(y(i)))
  end do
  write(*, '(A)') trim(line)
  write(*, '(A)') '# itol' // achar(9) // trim(integer_text(itol))
  line = '# rtol'
  do i = 1, merge(dimension_n, 1, itol == 1)
    line = trim(line) // achar(9) // trim(number(rtol(i)))
  end do
  write(*, '(A)') trim(line)
  line = '# atol'
  do i = 1, merge(dimension_n, 1, itol == 1)
    line = trim(line) // achar(9) // trim(number(atol(i)))
  end do
  write(*, '(A)') trim(line)
  write(*, '(A)') '# work(1)_uround' // achar(9) // trim(number(work(1))) // &
                  achar(9) // 'work(6)_hmax' // achar(9) // &
                  trim(number(work(6))) // achar(9) // 'work(7)_h0' // &
                  achar(9) // trim(number(work(7)))
  write(*, '(A)') '# iout 2, iwork(3)=-1 (silent; stiffness aborts with idid=-4),' // &
                  ' iwork(5)=n, all other work/iwork zero (defaults)'

#ifdef METHOD_DOP853
  call dop853(dimension_n, right_hand_side, x, y, x_end, rtol, atol, itol, &
              solution_output, iout, work, lwork, iwork, liwork, rpar, ipar, &
              idid)
#else
  call dopri5(dimension_n, right_hand_side, x, y, x_end, rtol, atol, itol, &
              solution_output, iout, work, lwork, iwork, liwork, rpar, ipar, &
              idid)
#endif

  write(*, '(A)') 'stats' // achar(9) // trim(integer_text(idid)) // &
    achar(9) // trim(integer_text(iwork(17))) // &
    achar(9) // trim(integer_text(iwork(18))) // &
    achar(9) // trim(integer_text(iwork(19))) // &
    achar(9) // trim(integer_text(iwork(20))) // &
    achar(9) // trim(number(x)) // achar(9) // trim(number(work(7)))
end program reference_driver
