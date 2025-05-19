# IV Curve Simulation Tool

This README is a guide to the usage of the IV (Current-Voltage) curve simulation program. This program calculates the dc current through superconducting junctions.

## Parameters

The program by itself takes the following parameters through the `iv.in` input file:

| Parameter | Description | Default |
|-----------|-------------|-----------------|
| `trans` | transmission coefficient: [0,1] | 0.5 |
| `temp` | Temperature in units of the gap at the corresponding temperature | 1e-7 |
| `wi` | Lower limit in the energy integration of the current (in units of the gap) | -15.000 |
| `wf` | Upper limit in the energy integration of the current (in units of the gap) | 15.00000 |
| `vi` | Initial voltage (in units of the gap) | 0.05 |
| `vf` | Final voltage (in units of the gap) | 3.51 |
| `vstep` | Voltage step (in units of the gap) | 0.01 |

## Running the Program

The program reads parameters from the `iv.in` file. To run the program and save the output to a file:

```
./iv > iv.dat
```

## Output Format

The output file contains two columns:
1. Voltage (in units of the gap)
2. Current (normalized by $\frac{2e\Delta}{h}$)

## Important Notes

1. This program only computes the dc current.

2. The program estimates the number of multiple Andreev reflections ($n$) with the condition: $n = \frac{2\Delta}{V_i}$. The maximum value allowed for $n_{max}=500$, meaning the minimum value allowed for the initial voltage is approximately $0.005\cdot\Delta$.

3. For energy integration, a custom integration subroutine is used with absolute tolerance ($Atol=1\cdot 10^{-8}$) and relative tolerance ($Rtol=1\cdot 10^{-6}$).

4. Energy integration range: Choose the lower limit (`wi`) and upper limit (`wf`) symmetrically. Use values that are at least several times larger than the maximum voltage (`vf`). At low voltages, you need to significantly expand the energy integration range to avoid getting negative current results.

5. For finite temperature calculations: All energy scales are normalized with the gap at the corresponding temperature. If you need the absolute value of the current, the temperature dependence of the gap can be implemented. 