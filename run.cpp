/**
 * @brief Orquestra o ciclo de vida completo da Maximização de Influência (IMM + Monte Carlo).
 * * Esta função atua como o controlador principal (entry-point da lógica). Ela é responsável por:
 * 1. Fazer o parse dos argumentos de entrada.
 * 2. Carregar a topologia do grafo para a memória de forma eficiente.
 * 3. Aplicar filtros de universo (se existirem).
 * 4. Invocar o motor matemático do algoritmo IMM para selecionar as sementes.
 * 5. Validar empiricamente o spread das sementes escolhidas via simulação de Monte Carlo.
 * * @param args Mapa associativo (std::map) contendo os argumentos de configuração do sistema.
 * Chaves esperadas: "graph", "k", "eps", "ell", "model", "numMC", e opcionalmente "universe".
 * * @note A função realiza chamadas pesadas de I/O e alocação dinâmica de memória. 
 * Ela utiliza o "Swap Trick" para garantir a desalocação do grafo bruto antes da fase de simulação.
 */
void run(map<string, string> args) {
	// ========================================================================
	// 1. PARSEAMENTO DE ARGUMENTOS E CONFIGURAÇÃO
	// ========================================================================
	/// Extrai os caminhos dos arquivos e converte os hiperparâmetros para seus tipos nativos.
	/// A função get_or_die() atua como um assert, abortando a execução se chaves obrigatórias faltarem.
	string input = get_or_die(args, "graph");
	int k = atoi(get_or_die(args, "k").c_str());         ///< Tamanho do budget de sementes (alvo de nós a selecionar)
	double eps = atof(get_or_die(args, "eps").c_str());  ///< Fator de erro tolerado (epsilon)
	double ell = atof(get_or_die(args, "ell").c_str());  ///< Fator de confiança probabilística
	string model = get_or_die(args, "model");            ///< Modelo de difusão: "tvic" ou "tvlt"
	int numMC = atoi(get_or_die(args, "numMC").c_str()); ///< Número de iterações para validação empírica

	// ========================================================================
	// 2. LEITURA DO GRAFO (I/O)
	// ========================================================================
	cout << "[1/4] Lendo arquivo..." << endl;
	ifstream is(input.c_str());
	if (!is.is_open()) {
		cerr << "Erro fatal: Nao foi possivel abrir " << input << endl;
		exit(1);
	}

	/// Estrutura temporária para armazenar a lista de adjacências bruta.
	vector<edge> ps;
	int V = 0; ///< Controlador dinâmico para descobrir o número total de vértices
	int u, v;
	double p_val;

	/// Lê o arquivo texto linha a linha no formato: [Origem] [Destino] [Probabilidade]
	while (is >> u >> v >> p_val) {
		/// Ignora auto-loops topológicos (u == v), pois não contribuem para o spread marginal.
		if (u == v) continue;

		edge e = { u, v, p_val };

		/// O número total de vértices é inferido pelo maior ID encontrado no arquivo + 1.
		V = max(V, max(u, v) + 1);
		ps.push_back(e);
	}
	is.close();

	// ========================================================================
	// 3. EXIBIÇÃO DE METADADOS
	// ========================================================================
	cout << "========================================" << endl;
	cout << "       IMM - CONFIGURACAO ATUAL" << endl;
	cout << "========================================" << endl;
	cout << "Dataset      : " << input << endl;
	cout << "Nos (V)      : " << V << endl;
	cout << "Arestas (E)  : " << ps.size() << endl;
	cout << "----------------------------------------" << endl;
	cout << "Modelo       : " << (model == "tvlt" ? "Linear Threshold (LT)" : "Independent Cascade (IC)") << endl;
	cout << "Sementes (k) : " << k << endl;
	cout << "Monte Carlo  : " << numMC << " simulacoes (Validacao)" << endl;
	cout << "========================================" << endl << endl;

	// ========================================================================
	// 4. CONSTRUÇÃO DAS ESTRUTURAS DO GRAFO E FILTRO DE UNIVERSO
	// ========================================================================
	cout << "[2/4] Construindo grafo (V=" << V << ", E=" << ps.size() << ")..." << endl;

	/// Vetor booleano que atua como máscara de elegibilidade. Se candidates[v] == true, 'v' pode ser selecionado.
	vector<bool> candidates(V, true); 
	string universe_file = "";

	/// Verifica a existência da chave "universe" no map de argumentos para aplicar a regra de restrição.
	if (args.count("universe")) {
		universe_file = args["universe"];
		cout << "[Info] Arquivo de universo detectado: " << universe_file << endl;

		ifstream u_file(universe_file.c_str());
		if (u_file.is_open()) {
			/// Invalida toda a rede primeiro, para depois habilitar estritamente os nós listados.
			fill(candidates.begin(), candidates.end(), false);

			int node_id;
			int count_valid = 0;
			while (u_file >> node_id) {
				if (node_id >= 0 && node_id < V) {
					candidates[node_id] = true;
					count_valid++;
				}
			}
			u_file.close();
			cout << "[Info] Universo restrito a " << count_valid << " nos." << endl;
		}
		else {
			cout << "[Aviso] Arquivo de universo nao encontrado. Usando todos os nos." << endl;
		}
	}
	else {
		cout << "[Info] Nenhum universo especificado. Busca global." << endl;
	}

	/// @var rs (Reverse Structure): Lista de adjacência transposta (v -> u). Fundamental para geração dos RR-Sets.
	vector<vector<edge> > rs(V);

	/// @var es (Edge Structure): Lista de adjacência direta (u -> v). Utilizada estritamente na validação Monte Carlo.
	vector<vector<edge> > es(V);

	/// Popula as estruturas de adjacência de forma simultânea.
	for (auto e : ps) {
		rs[e.v].push_back(e); 
		es[e.u].push_back(e); 
	}

	/// Aplicação do 'Swap Trick': Força o compilador a desalocar a memória heap ocupada pelo vetor bruto 'ps'.
	/// Vital para evitar Out-Of-Memory (OOM) em grafos de larga escala.
	{
		vector<edge> empty;
		ps.swap(empty);
	}

	// ========================================================================
	// 5. EXECUÇÃO DO ALGORITMO IMM (CORE)
	// ========================================================================
	cout << "[3/4] Executando IMM (Selecao de Sementes)..." << endl;
	clock_t start_imm = clock();

	/// Invoca o motor matemático para encontrar as sementes ótimas, passando o grafo reverso e a máscara de candidatos.
	vector<int> S = imm(V, rs, model, k, eps, ell, candidates);

	clock_t end_imm = clock();

	cout << "\n>>> TEMPO IMM: " << (double)(end_imm - start_imm) / CLOCKS_PER_SEC << "s <<<" << endl;
	cout << "Sementes Escolhidas: { ";
	for (size_t i = 0; i < S.size(); i++) cout << S[i] << (i < S.size() - 1 ? ", " : "");
	cout << " }" << endl;

	// ========================================================================
	// 6. VALIDAÇÃO VIA MONTE CARLO (PÓS-PROCESSAMENTO)
	// ========================================================================
	cout << "\n[4/4] Validacao Monte Carlo (" << numMC << " simulacoes)..." << endl;

	clock_t start_mc = clock();

	/// Acumuladores de estado globais, manipulados em contexto multithread.
	double total_inf = 0.0;             
	double global_running_spread = 0.0; 
	int progress_counter = 0;           

	/// Diretiva OpenMP: Inicializa um time de threads para processamento em paralelo.
#pragma omp parallel
	{
		/// Garante que cada thread possua uma semente única (XOR do PID da thread) para o motor Mersenne Twister.
		unsigned long seed = (unsigned long)(time(NULL) ^ (omp_get_thread_num() * 99999));
		mt19937 gen(seed);

		double local_inf = 0; ///< Acumulador local para evitar falsos compartilhamentos (false sharing) em cache.

		/// Divide dinamicamente as iterações do loop for entre as threads disponíveis.
#pragma omp for
		for (int sim = 0; sim < numMC; sim++) {
			int result = 0;

			/// Executa a caminhada aleatória (forward) baseada no modelo selecionado.
			if (model == "tvic" || model == "ic") {
				result = MonteCarlo_IC(V, es, S, gen);
			}
			else if (model == "tvlt" || model == "lt") {
				result = MonteCarlo_LT(V, es, S, gen);
			}

			local_inf += result;

			/// Operações atômicas garantem thread-safety (mutex a nível de hardware) para atualizar os contadores globais.
#pragma omp atomic
			global_running_spread += result;

#pragma omp atomic
			progress_counter++;

			/// Controle de renderização do console: Atualiza a barra de progresso a cada 100 iterações.
			if (progress_counter % 100 == 0 || progress_counter == numMC) {
				/// Seção Crítica: Impede que o fluxo de saída (stdout) seja mesclado por múltiplas threads simultaneamente.
#pragma omp critical
				{
					double current_avg = global_running_spread / progress_counter;
					cout << "\rProgresso: " << progress_counter << "/" << numMC
						<< " (" << (int)(100.0 * progress_counter / numMC) << "%) "
						<< "| Spread Est.: " << fixed << setprecision(2) << current_avg << "   " << flush;
				}
			}
		}

		/// Ao final do bloco paralelo de cada thread, consolida o resultado local na variável global de retorno.
#pragma omp atomic
		total_inf += local_inf;
	}

	clock_t end_mc = clock();

	// ========================================================================
	// 7. RESULTADOS FINAIS E SAÍDA
	// ========================================================================
	cout << "\n\nCalculo Finalizado." << endl;
	double final_spread = total_inf / numMC;

	cout << "========================================" << endl;
	cout << "SPREAD MEDIO FINAL: " << final_spread << endl;
	cout << "TEMPO MONTE CARLO : " << (double)(end_mc - start_mc) / CLOCKS_PER_SEC << "s" << endl;
	cout << "========================================" << endl;

	cout << "Pressione ENTER para sair..." << endl;
	cin.get();
}