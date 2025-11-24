module eigendecomp_interface
    use iso_c_binding
    implicit none

    interface
        subroutine eigenDecompSolver_float(frontBack, fbLd, leftRight, lrLd, &
                topBottom, tbLd, f, fStride, x, xStride, &
                height, width, depth) bind(C, name="eigendecompsolver_float_")
            integer(c_size_t), value :: fbLd, lrLd, tbLd
            integer(c_size_t), value :: fStride, xStride
            integer(c_size_t), value :: height, width, depth
            real(c_float), intent(in) :: frontBack(*), leftRight(*), topBottom(*)
            real(c_float), intent(out) :: f(*), x(*)
        end subroutine eigenDecompSolver_float

        subroutine eigenDecompSolver_double(frontBack, fbLd, leftRight, lrLd, &
                topBottom, tbLd, f, fStride, x, xStride, &
                height, width, depth) bind(C, name="eigendecompsolver_double_")
            integer(c_size_t), value :: fbLd, lrLd, tbLd
            integer(c_size_t), value :: fStride, xStride
            integer(c_size_t), value :: height, width, depth
            real(c_double), intent(in) :: frontBack(*), leftRight(*), topBottom(*)
            real(c_double), intent(out) :: f(*), x(*)
        end subroutine eigenDecompSolver_double
    end interface
end module eigendecomp_interface
