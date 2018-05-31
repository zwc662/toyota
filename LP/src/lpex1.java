import ilog.concert.*;
import ilog.cplex.*;
import java.util.*;
public class lpex1 {
	public static void main(String[] args) {
		try {
			int n = 10000;
			if(args.length > 1) n = Integer.parseInt(args[1]);
			
			Random r = new Random();
			
			IloCplex cplex = new IloCplex();
			double[] lb = new double[n];
			Arrays.fill(lb, -100);
			double[] ub = new double[n];
			Arrays.fill(ub, 100);
			IloNumVar[] x = cplex.numVarArray(n, lb, ub);
			
			double[] objvals = new double[n];
			for(int i = 0; i < n; i++) objvals[i] = r.nextFloat();
			cplex.addMaximize(cplex.scalProd(x, objvals));
			
			IloNumExpr temp;
			
			for(int i = 0; i < n; i++) {
				temp = cplex.prod(r.nextFloat(), x[0]);
				for(int j = 1; j < n; j++) {
					temp = cplex.sum(temp, cplex.prod(r.nextFloat(), x[j]));
				}
				cplex.addLe(temp, r.nextFloat());
			}
			
			if ( cplex.solve() ) {
				cplex.output().println("Solution status = " + cplex.getStatus());
				cplex.output().println("Solution value = " + cplex.getObjValue());
				double[] val = cplex.getValues(x);
				int ncols = cplex.getNcols();
				for (int j = 0; j < ncols; ++j)
					cplex.output().println("Column: " + j + " Value = " + val[j]);
			}
			cplex.end();
		}
		catch (IloException e) {
			System.err.println("Concert exception ’" + e + "’ caught");
		}
	}
}
