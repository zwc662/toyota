//==============================================================================
//	
//	Copyright (c) 2017-
//	Authors:
//	* Dave Parker <d.a.parker@cs.bham.ac.uk> (University of Birmingham)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================

package demos;

import java.io.FileOutputStream;
import java.io.PrintStream;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import parser.ast.*;
import prism.PrismLangException;
import prism.PrismUtils;
import parser.type.*;

import prism.ModelType;
import prism.Prism;
import prism.PrismDevNullLog;
import prism.PrismException;
import prism.PrismLog;
import prism.Result;
import prism.UndefinedConstants;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import Jama.Matrix;

/**
 * An example class demonstrating how to control PRISM programmatically, through
 * the functions exposed by the class prism.Prism.
 * 
 * This shows how to load a model from a file and model check some properties,
 * either from a file or specified as a string, and possibly involving
 * constants.
 * 
 * See the README for how to link this to PRISM.
 */
public class grid_world_v2 {
	public static int x_max = 0;
	public static int y_max = 0;
	public static int actions = 0;
	public static Matrix P_OPT;
	public static Matrix policy;
	
	public static void main(String[] args) throws IOException, InterruptedException, PrismLangException {
		new grid_world_v2().run();
	}

	static final public void ConstantDef(ConstantList constantList, ArrayList<String> lines) {
		String sLastLine = lines.get(0), sCurrentLine = lines.get(1);
		for (String line : lines) {
			if (lines.indexOf(line) % 2 == 1) {
				sCurrentLine = line;
				try {
					if (sLastLine.equals("x_max"))
						x_max = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("y_max"))
						y_max = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("actions"))
						actions = Integer.parseInt(sCurrentLine);
					constantList.addConstant(new ExpressionIdent(sLastLine),
							new ExpressionLiteral(TypeInt.getInstance(), Integer.parseInt(sCurrentLine)),
							TypeInt.getInstance());
				} catch (NumberFormatException e) {
					constantList.addConstant(new ExpressionIdent(sLastLine),
							new ExpressionLiteral(TypeDouble.getInstance(), Double.parseDouble(sCurrentLine)),
							TypeDouble.getInstance());
				}
			} else {
				sLastLine = line;
			}
		}
	}

	static final public void ParsePolicy(ArrayList<String> lines) {
		policy = new Matrix(new double[y_max * x_max][y_max * x_max]);
		for (int i = 0; i < lines.size(); i++) {
			String[] line = lines.get(i).split(" ");
			policy.set(Integer.parseInt(line[0]), Integer.parseInt(line[1]), Double.parseDouble(line[2]));
		}
	}


	static final public Module Module(String name, ConstantList constantList, FormulaList formulaList) {
		Module m = new Module(name);
		m.setName(name);
		m.addDeclaration(new Declaration("x", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0),
				new ExpressionLiteral(TypeInt.getInstance(), x_max-1))));
		m.addDeclaration(new Declaration("y", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0),
				new ExpressionLiteral(TypeInt.getInstance(), y_max-1))));
		build_cmd(m);
		return m;
	}

	static final public void build_cmd(Module m) {
		for (int i = 0; i < policy.getRowDimension(); i++) {
		    int y = (int)i/x_max;
		    int x = (int)i%x_max;
			Command c = new Command();
			Updates us = new Updates();
			Update u = new Update();
			c.setGuard(new ExpressionLiteral(TypeBool.getInstance(),  "(x= "+ x + "& y= " + y + " = true"));
			for (int j = 0; j < policy.getColumnDimension(); j++) {
				double p = policy.get(i, j);
				if(p > 0.0) {
					int y_ = (int)j/x_max;
				    int x_ = (int)j%x_max;
					u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), Integer.toString(x_)));
					u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), Integer.toString(y_)));
					us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), Double.toString(p)), u);
					u = new Update();
				}
			}
			c.setUpdates(us);
			m.addCommand(c);
		} 
	}

	
	static final public void run() throws InterruptedException, FileNotFoundException {
		try {
			// Create a log for PRISM output (hidden or stdout)
			PrismLog mainLog = new PrismDevNullLog();
			// PrismLog mainLog = new PrismFileLog("stdout");

			// Initialise PRISM engine
			Prism prism = new Prism(mainLog);
			prism.initialise();

			ModulesFile mf = new ModulesFile();

			mf.setModelType(ModelType.DTMC);

			ArrayList<String> files = new ArrayList<String>();
			String STATE_SPACE = "//home/zekunzhou/workspace/Safety-AI-MDP/mountaincar/state_space";
			String OPT_POLICY = "//home/zekunzhou/workspace/Safety-AI-MDP/mountaincar/optimal_policy";
			files.add(STATE_SPACE);
			files.add(OPT_POLICY);
			ArrayList<String> lines = new ArrayList<String>();
			for (String file : files) {
				BufferedReader br = null;
				FileReader fr = null;
				try {
					fr = new FileReader(file);
					br = new BufferedReader(fr);
					String line;
					br = new BufferedReader(new FileReader(file));
					while ((line = br.readLine()) != null) {
						lines.add(line);
					}
					if (file.equals(STATE_SPACE)) {
						ConstantDef(mf.getConstantList(), lines);
						lines.clear();
					}
					if (file.equals(OPT_POLICY)) {
						ParsePolicy(lines);
						lines.clear();
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			Module m_opt = Module("grid_world", mf.getConstantList(), mf.getFormulaList());
			mf.addModule(m_opt);
			mf.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "x = 0 & y = 0"));

			mf.tidyUp();
			
			prism.loadPRISMModel(mf);

			PrintStream ps_console = System.out;
			PrintStream ps_file = new PrintStream(new FileOutputStream(
					new File("//home/zekunzhou/workspace/Safety-AI-MDP/mountaincar/grid_world.pm")));
			System.setOut(ps_file);
			System.out.println(mf);
			
			System.setOut(ps_console);
			
			System.exit(1);
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		} catch (PrismException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		}

	}
}