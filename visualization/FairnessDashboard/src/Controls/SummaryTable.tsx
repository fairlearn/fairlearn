import React from "react";
import { List } from "office-ui-fabric-react/lib/List";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { Text } from "office-ui-fabric-react/lib/Text";
import { mergeStyleSets } from "@uifabric/styling";
import { Separator } from "office-ui-fabric-react/lib/Separator";

export interface ISummaryTableProps {
    binValues: number[];
    binLabels: string[];
}

interface IBinItem {
    title: string;
    score: number;
    isMin: boolean;
    isMax: boolean;
}

export class SummaryTable extends React.PureComponent<ISummaryTableProps> {

    private static separatorStyle = {
        root: [{
          selectors: {
            '::after': {
              backgroundColor: 'darkgrey',
            },
          }
        }]
    };

    private static readonly classNames = mergeStyleSets({
        itemCell: [
          {
            display: "inline-flex",
            width: "100%",
            position: "relative",
            float: "left",
            boxSizing: "border-box",
            border: `1px solid grey`,
            height: "100%"
          }
        ],
        itemsList: {
            paddingTop: "20px",
            paddingBottom: "57px",
            boxSizing: "border-box",
            height: "100%",
            width: "300px",
            selectors: {
                ".ms-List-surface": {
                    height: "100%"
                },
                ".ms-List-page": {
                    height: "100%",
                    display: "flex",
                    flexDirection: "column"
                },
                ".ms-List-cell": {
                    flex: 1
                }
            }
        },
        minMaxLabel: {
            padding: "2px, 5px",
            backgroundColor: "#CCC"
        }
    });
    
    public render(): React.ReactNode {
        let minIndexes = [];
        let maxIndexes = [];
        let minValue = Number.MAX_SAFE_INTEGER;
        let maxValue = Number.MIN_SAFE_INTEGER;
        this.props.binValues.forEach((value, index) => {
            if (value >= maxValue) {
                if (value === maxValue) {
                    maxIndexes.push(index);
                } else {
                    maxIndexes = [index];
                    maxValue = value;
                }
            }
            if (value <= minValue) {
                if (value === minValue) {
                    minIndexes.push(index);
                } else {
                    minIndexes = [index];
                    minValue = value;
                }
            }
        });
        return (
            <List className={SummaryTable.classNames.itemsList}
                    items={this.props.binValues.map((binValue, index) => {
                                    return {
                                        title: this.props.binLabels[index],
                                        score: binValue,
                                        isMin: minIndexes.includes(index),
                                        isMax: maxIndexes.includes(index)
                                    };
                                })}
                    onRenderCell={this._onRenderCell}
            />);
    }

    private readonly _onRenderCell = (item: IBinItem, index: number | undefined): JSX.Element => {
        return (
          <div
            key={index}
            className={SummaryTable.classNames.itemCell}
            >
            <div style={{width: "48%", alignSelf: "center"}}>
                <Stack verticalAlign="space-evenly">
                    <div>
                        <div>
                            <Text style={{fontSize: "25px"}}>{item.title}</Text>
                        </div>
                        <Stack horizontal>
                            {item.isMin && <div className={SummaryTable.classNames.minMaxLabel}>Min</div>}
                            {item.isMax && <div className={SummaryTable.classNames.minMaxLabel}>Max</div>}
                        </Stack>
                    </div>
                </Stack>
            </div>
            <Separator vertical styles={SummaryTable.separatorStyle} />
            <div style={{flex: "1", alignSelf: "center"}}> 
                <Stack verticalAlign="space-evenly">
                    <Text style={{fontSize:"55px", paddingLeft:"10px"}}>{item.score.toLocaleString(undefined, {style: "percent", maximumFractionDigits: 1})}</Text>
                </Stack>
            </div>
          </div>
        );
    }
}