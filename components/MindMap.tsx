import React, { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { MindMapData, MindMapNode, MindMapLink, MindMapNodeType } from '../types';
import { BrainIcon } from './icons/BrainIcon';

const getNodeColor = (type: MindMapNodeType) => {
    switch(type) {
        case 'CORE_PERSONA': return '#06b6d4'; // cyan-500
        case 'MISSION': return '#a855f7'; // purple-500
        case 'TASK': return '#64748b'; // slate-500
        case 'PSYCHOLOGY_ASPECT': return '#6366f1'; // indigo-500
        case 'QUANTUM_INSIGHT': return '#ec4899'; // pink-500
        case 'KEY_TRAIT': return '#eab308'; // yellow-500
        case 'STRENGTH': return '#84cc16'; // lime-500
        case 'WEAKNESS': return '#f97316'; // orange-500
        case 'KNOWLEDGE_CONCEPT': return '#3b82f6'; // blue-500
        case 'ABSTRACT_CONCEPT': return '#10b981'; // emerald-500
        case 'FILE_REFERENCE': return '#78716c'; // stone-500
        default: return '#6b7280'; // gray-500
    }
}

interface MindMapGraphProps {
  data: MindMapData;
  onNodeClick: (node: MindMapNode | null) => void;
  selectedNodeId: string | null;
}

export const MindMapGraph: React.FC<MindMapGraphProps> = ({ data, onNodeClick, selectedNodeId }) => {
    const svgRef = useRef<SVGSVGElement>(null);
    const simulationRef = useRef<d3.Simulation<MindMapNode, MindMapLink>>();
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();

    const memoizedData = useMemo(() => {
        // Deep copy to prevent mutation of props
        return JSON.parse(JSON.stringify(data));
    }, [data]);

    useEffect(() => {
        if (!svgRef.current) return;

        const svg = d3.select(svgRef.current);
        const g = svg.select<SVGGElement>("g.everything");
        const width = svg.node()!.getBoundingClientRect().width;
        const height = svg.node()!.getBoundingClientRect().height;

        // Initialize simulation if it doesn't exist
        if (!simulationRef.current) {
            simulationRef.current = d3.forceSimulation<MindMapNode, MindMapLink>()
                .force("link", d3.forceLink<MindMapNode, MindMapLink>().id((d: any) => d.id).distance(d => d.type === 'HIERARCHICAL' ? 80 : 120).strength(0.8))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX(width / 2).strength(0.05))
                .force("y", d3.forceY(height / 2).strength(0.05));
        }
        
        const simulation = simulationRef.current;
        
        let link = g.select<SVGGElement>(".links").selectAll<SVGPathElement, MindMapLink>("path");
        let node = g.select<SVGGElement>(".nodes").selectAll<SVGGElement, MindMapNode>("g.node-group");

        // --- Data Binding ---
        link = link.data(memoizedData.links, (d: any) => `${d.source.id || d.source}-${d.target.id || d.target}`);
        node = node.data(memoizedData.nodes, (d: any) => d.id);

        // --- Exit ---
        link.exit().transition().duration(300).attr("stroke-opacity", 0).remove();
        node.exit().transition().duration(300).attr("opacity", 0).remove();

        // --- Enter ---
        const linkEnter = link.enter().append("path")
            .attr("stroke", "#475569") // slate-600
            .attr("stroke-width", 1.5)
            .attr("fill", "none")
            .attr("stroke-opacity", 0);
        
        linkEnter.transition().duration(300)
            .attr("stroke-opacity", 0.6);
            
        const nodeEnter = node.enter().append("g")
            .attr("class", "node-group")
            .attr("cursor", "pointer")
            .on("click", (event, d) => {
              event.stopPropagation(); // prevent zoom on click
              onNodeClick(d);
            })
            .call(drag(simulation) as any);

        nodeEnter.append("circle")
            .attr("r", d => d.type === 'CORE_PERSONA' ? 14 : d.type === 'MISSION' ? 12 : 9)
            .attr("stroke", "#111827") // slate-900
            .attr("stroke-width", 2);

        nodeEnter.append("text")
            .text(d => d.name)
            .attr("x", d => d.type === 'CORE_PERSONA' ? 18 : d.type === 'MISSION' ? 16 : 13)
            .attr("y", 5)
            .attr("fill", "#e5e7eb") // gray-200
            .style("font-size", "12px")
            .style("pointer-events", "none")
            .style("text-shadow", "0 0 3px #000, 0 0 3px #000");
        
        nodeEnter.attr("opacity", 0)
            .transition().duration(300)
            .attr("opacity", 1);


        // --- Merge (Update) ---
        link = linkEnter.merge(link);
        node = nodeEnter.merge(node);
        
        node.select("circle")
            .attr("fill", d => getNodeColor(d.type))
            .transition().duration(200)
            .attr("stroke", d => d.id === selectedNodeId ? "#2dd4bf" : "#111827") // teal-400 for selected
            .attr("stroke-width", d => d.id === selectedNodeId ? 3.5 : 2);


        simulation.nodes(memoizedData.nodes);
        (simulation.force("link") as d3.ForceLink<MindMapNode, MindMapLink>).links(memoizedData.links);

        simulation.on("tick", () => {
             link.attr("d", (d: any) => {
                if(!d.source.x || !d.target.x) return null; // handle exit case
                return `M${d.source.x},${d.source.y} L${d.target.x},${d.target.y}`;
             });
            node.attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
        });

        simulation.alpha(0.8).restart();

        // --- Drag functionality ---
        function drag(simulation: d3.Simulation<MindMapNode, MindMapLink>) {
            function dragstarted(event: any, d: any) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            function dragged(event: any, d: any) {
                d.fx = event.x;
                d.fy = event.y;
            }
            function dragended(event: any, d: any) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
            return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
        }
        
         // --- Zoom functionality ---
        if (!zoomRef.current) {
            zoomRef.current = d3.zoom<SVGSVGElement, unknown>()
                .scaleExtent([0.1, 4])
                .on("zoom", (event) => {
                    g.attr('transform', event.transform);
                });
            svg.call(zoomRef.current);
        }

    }, [memoizedData, selectedNodeId, onNodeClick]);


    return (
        <svg ref={svgRef} className="w-full h-full">
            <g className="everything">
                <g className="links"></g>
                <g className="nodes"></g>
            </g>
        </svg>
    );
};


export const MindMap: React.FC<MindMapGraphProps> = ({ data, onNodeClick, selectedNodeId }) => {
  return (
    <div className="w-full h-full bg-gray-900">
       {data.nodes.length > 0 ? (
        <MindMapGraph data={data} onNodeClick={onNodeClick} selectedNodeId={selectedNodeId} />
      ) : (
        <div className="w-full h-full flex items-center justify-center text-gray-600">
          <div className="text-center">
            <BrainIcon className="w-24 h-24 mx-auto text-cyan-900" />
            <p className="mt-4 text-lg font-orbitron">KNOWLEDGE GRAPH WILL BE GENERATED HERE</p>
            <p className="text-sm text-gray-500">Define a persona and click 'Create Mind' to begin.</p>
          </div>
        </div>
      )}
    </div>
  );
};