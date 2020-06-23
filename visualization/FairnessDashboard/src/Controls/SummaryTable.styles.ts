import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights } from 'office-ui-fabric-react';

export interface ISummaryTableStyles {
    minMaxLabel: IStyle;
    groupCol: IStyle;
    groupLabel: IStyle;
    flexCol: IStyle;
    binBox: IStyle;
    binTitle: IStyle;
    metricCol: IStyle;
    metricBox: IStyle;
    metricLabel: IStyle;
    frame: IStyle;
}

export const SummaryTableStyles: () => IProcessedStyleSet<ISummaryTableStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ISummaryTableStyles>({
        minMaxLabel: {
            padding: '1px 9px',
            marginTop: '4px',
            color: theme.semanticColors.bodySubtext,
            backgroundColor: theme.semanticColors.bodyStandoutBackground,
        },
        groupCol: {
            display: 'inline-flex',
            flexDirection: 'column',
            height: '100%',
            width: 'max-content',
        },
        groupLabel: {
            color: theme.semanticColors.bodyText,
            height: '26px',
        },
        flexCol: {
            display: 'flex',
            flex: 1,
            flexDirection: 'column',
            borderTop: '1px solid',
            borderBottom: '1px solid',
            borderColor: theme.semanticColors.bodyDivider,
        },
        binBox: {
            flex: 1,
            width: '130px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            borderBottom: '0.5px dashed',
            borderColor: theme.semanticColors.bodyDivider,
        },
        binTitle: {
            color: theme.semanticColors.bodyText,
        },
        metricCol: {
            display: 'inline-flex',
            flexDirection: 'column',
            height: '100%',
            width: '120px',
        },
        metricLabel: {
            color: theme.semanticColors.bodyText,
            height: '26px',
            paddingLeft: '10px',
        },
        metricBox: {
            flex: 1,
            paddingLeft: '10px',
            color: theme.semanticColors.bodyText,
            fontWeight: FontWeights.light,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            borderLeft: '0.5px dashed',
            borderBottom: '0.5px dashed',
            borderColor: theme.semanticColors.bodyDivider,
        },
        frame: {
            paddingBottom: '19px',
            display: 'flex',
        },
    });
};
