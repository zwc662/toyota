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
public class grid_world {
	public static int states = 0;
	public static int actions = 0;
	public static Matrix P_OPT;
	public static double[][] policy;
	public static double[][] unsafe;
	public static int TRANSITIONS = 0;
	public static int STATES = 0;
	public static ArrayList<String> dtmc; 
	public static void main(String[] args) throws IOException, InterruptedException, PrismLangException {
		System.out.println(args[0]);
		new grid_world().run(args[0]);
	}

	static final public void ConstantDef(ConstantList constantList, ArrayList<String> lines) {
		String sLastLine = lines.get(0), sCurrentLine = lines.get(1);
		for (String line : lines) {
			if (lines.indexOf(line) % 2 == 1) {
				sCurrentLine = line;
				try {
					if (sLastLine.equals("states"))
						states = Integer.parseInt(sCurrentLine);
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

	static final public void ParsePolicy(Module m, ArrayList<String> lines) {
		policy = new double[states][1];
		int from = 0;
		for (int i = 0; i < lines.size(); i++) {
			String[] line = lines.get(i).split(" ");
			from = Integer.parseInt(line[0]);
			policy[Integer.parseInt(line[1])][0] = (1.0 - unsafe[Integer.parseInt(line[0])][0]) * Double.parseDouble(line[2]);
		}
		for (int i = 0; i <= states - 1; i++) {
			policy[states - 1][0] =  unsafe[i][0];
		}
		Check_Transitions(from, policy);
		build_cmd(m, policy, from);
	}
	
	static final public void ParseUnsafe(ArrayList<String> lines) {
		unsafe = new double[states][1];
		for (int i = 0; i < lines.size(); i++) {
			String[] line = lines.get(i).split(":");
			unsafe[Integer.parseInt(line[0])][0] = 1.0;
		}
		unsafe[states - 1][0] = 1.0; 
	}

	static final public void Normalize_Transitions(int from, double[][] policy, double p_total) {
		if(p_total < 1.0) {
			Double p = policy[from][0];
			policy[from][0] = p + 1.0 - p_total;// - 0.0001);
			System.out.println("Row " + Integer.toString(from) + " Operation Complete");
		}
		else if (p_total > 1.0) {
			for(int j = 0; j < policy.length; j++) {
				/*
				Double p = policy[from][0];
				if(p > p_total - 1.0 + 0.0001) {
					policy[from][0] = p - (p_total - 1.0 + 0.0001);
					p_total = p_total - (p_total - 1.0 + 0.0001);
					System.out.println("Row " + Integer.toString(from) + " Operation Complete");
					break;	
				}
				else {
					policy[from][0] = 0.0;
					p_total = p_total - p;
				}
				*/
				policy[j][0] /= p_total;
			}
		}	
	}
	
	static final public void Check_Transitions(int from, double[][] policy) {
		Boolean recheck = true;
		while(recheck) {
			Double p_total = 0.0;
			for(int j = 0; j < policy.length; j++) {
				p_total = p_total + policy[j][0]; 
			}
			if(p_total != 1.0) {
				System.out.println("Row " + Integer.toString(from) + " prob sum is not 1.0 but " + Double.toString(p_total));
				Normalize_Transitions(from, policy, p_total);
			}
			else
				recheck = false;
			
			recheck = false;
		}
				
	}
	

	static final public Module Module(String name, ConstantList constantList, FormulaList formulaList) {
		Module m = new Module(name);
		m.setName(name);
		m.addDeclaration(new Declaration("s", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0),
				new ExpressionLiteral(TypeInt.getInstance(), states - 1))));
		return m;
	}

	static final public void build_cmd(Module m, double[][] policy, int s) {
		dtmc = new ArrayList<String>();
		double p_total = 0.0;
		int i = s;
		Command c = new Command();
		Updates us = new Updates();
		Update u = new Update();
		c.setGuard(new ExpressionLiteral(TypeBool.getInstance(),  "(s= "+ s + ") = true"));
		for (int j = 0; j < policy.length; j++) {
			double p = policy[j][0];
			if(p > 0.0) {
				p_total += p;
			    int s_ = (int)j;
				u.addElement(new ExpressionIdent("s"), new ExpressionLiteral(TypeInt.getInstance(), Integer.toString(s_)));
				us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), Double.toString(p)), u);
				TRANSITIONS = TRANSITIONS + 1;
				dtmc.add(Integer.toString(i) + ' ' + Integer.toString(j) + ' ' + Double.toString(p * 0.99999));
				u = new Update();
			}
		}
		c.setUpdates(us);
		m.addCommand(c);
	}
	
	static final public void Write_DTMC() {
		dtmc.add(0, "STATES " + Integer.toString(STATES));
		dtmc.add(1, "TRANSITIONS " + Integer.toString(TRANSITIONS));
		dtmc.add(2, "INITIAL " + Integer.toString(policy.length- 2));
		dtmc.add(3, "TARGET " + Integer.toString(policy.length - 1));
	}
	
	static final public void run(String path) throws InterruptedException, FileNotFoundException {
		try {
			// Create a log for PRISM output (hidden or stdout)
			PrismLog mainLog = new PrismDevNullLog();
			// PrismLog mainLog = new PrismFileLog("stdout");

			// Initialise PRISM engine
			Prism prism = new Prism(mainLog);
			prism.initialise();

			ModulesFile mf = new ModulesFile();

			mf.setModelType(ModelType.DTMC);
			Module m_opt;
			ArrayList<String> files = new ArrayList<String>();
			String STATE_SPACE = path + "/state_space";
			String OPT_POLICY = path + "/optimal_policy";
			String UNSAFE = path + "/unsafe";
			
			files.add(STATE_SPACE);
			files.add(UNSAFE);
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
					
					if (file.equals(STATE_SPACE)) {
						while ((line = br.readLine()) != null) {
							lines.add(line);
						}
						ConstantDef(mf.getConstantList(), lines);
						lines.clear();
					}
					if (file.equals(UNSAFE)) {
						while ((line = br.readLine()) != null) {
							lines.add(line);
						}
						ParseUnsafe(lines);
						lines.clear();
					}
					
					
					if (file.equals(OPT_POLICY)) {
						m_opt = Module("grid_world", mf.getConstantList(), mf.getFormulaList());
						int episode = states;
						while ((line = br.readLine()) != null) {
							if(episode == 0) {
								episode = states;
								ParsePolicy(m_opt, lines);
								lines.clear();	
							}
							lines.add(line);
							episode--;
							}
						mf.addModule(m_opt);		
					 }
					
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			//Check_Transitions();
			
			
			
			Write_DTMC();
			//mf.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "s = 0 & y = 0"));
			mf.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(),
					"s = " + Integer.toString(0)));
			mf.tidyUp();
			//System.out.println(mf);
			prism.loadPRISMModel(mf);
			PropertiesFile pf = prism.parsePropertiesString(mf, "P=? [F s=" + Integer.toString(states - 1)+ "]");

			PrintStream ps_console = System.out;
			PrintStream ps_file = new PrintStream(new FileOutputStream(
					new File(path + "/grid_world.pm")));
			System.setOut(ps_file);
			System.out.println(mf);
			
			PrintStream pctl_file = new PrintStream(new FileOutputStream(
					new File(path + "/grid_world.pctl")));
			System.setOut(pctl_file);
			System.out.println(pf);
			
			PrintStream dtmc_file = new PrintStream(new FileOutputStream(
					new File(path + "/grid_world.dtmc")));
			System.setOut(dtmc_file);
			for(String i:dtmc) {
				System.out.println(i);
			}
			
			System.setOut(ps_console);
			/**
			System.out.println(mf);
			PropertiesFile pf = prism.parsePropertiesFile(mf,
					new File(path + "/grid_world.pctl"));
			Result result = prism.modelCheck(pf, pf.getPropertyObject(0));
			System.out.println(result.getResult());
			**/

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