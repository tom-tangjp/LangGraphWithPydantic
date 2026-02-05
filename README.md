 pipreqs ./ --force --ignore .venv,tmp
 

    # Mermaid CLI（需要 Node.js）                                                                                       
    npm install -g @mermaid-js/mermaid-cli                                                                              
                                                                                                                        
    # Python 图表库                                                                                                     
    pip install diagrams networkx matplotlib                                                                            
                                                                                                                        
    # Graphviz                                                                                                          
    brew install graphviz  # macOS                                                                                      
    apt-get install graphviz  # Lin


                                                                                                                        
  注意：部分工具需要系统安装对应命令：                                                                                  
                                                                                                                        
  • clang-tidy、clang++、clang-format（LLVM/Clang 工具链）                                                              
  • cppcheck（C++ 静态分析器